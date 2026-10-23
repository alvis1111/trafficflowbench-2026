"""Quick probe: does spatial (up/downstream neighbour) context beat temporal-only?

On a few days of one panel, compare RMSE of temporal interpolation vs spatial
neighbour average vs a least-squares blend, on the masked cells (where we know
the truth from the unmasked layer).
"""
import glob
import os
import numpy as np
import pandas as pd

PANEL = "D12_I405_S"
ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public/corridors/" + PANEL

# --- neighbour map from topology ---
topo = pd.read_csv(f"{ROOT}/network/lwr_mainline_topology.csv", dtype=str)
def main_ids(s):
    return [x for x in str(s).split(";") if x and x.startswith("L")] if pd.notna(s) else []
neighbors = {}
for _, r in topo.iterrows():
    neighbors[r["mainline_link_id"]] = main_ids(r.get("incoming_link_ids")) + main_ids(r.get("outgoing_link_ids"))
n_main = len([k for k, v in neighbors.items() if v])
print(f"mainline links with neighbours: {n_main}/{len(neighbors)}")

# --- process a few days ---
masked_files = sorted(glob.glob(f"{ROOT}/train/mainline_states_masked/mask_regime=R*/*.parquet"))
truth_files = {os.path.basename(p): p for p in glob.glob(f"{ROOT}/train/mainline_states/year_month=*/*.parquet")}

rows = []
for mf in masked_files[:12]:
    name = os.path.basename(mf)
    tf = truth_files.get(name)
    if tf is None:
        continue
    m = pd.read_parquet(mf, columns=["timestamp", "link_id", "speed_kmh", "flow_vph"])
    t = pd.read_parquet(tf, columns=["timestamp", "link_id", "speed_kmh", "flow_vph"])
    m["link_id"] = m.link_id.astype(str)
    t["link_id"] = t.link_id.astype(str)
    m = m.sort_values(["link_id", "timestamp"]).reset_index(drop=True)

    # temporal interp per link
    m["ti_speed"] = m.groupby("link_id")["speed_kmh"].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both"))
    m["ti_flow"] = m.groupby("link_id")["flow_vph"].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both"))

    # spatial: pivot observed -> link x time, average neighbours
    sp = m.pivot_table(index="timestamp", columns="link_id", values="speed_kmh")
    fl = m.pivot_table(index="timestamp", columns="link_id", values="flow_vph")
    sp_est = pd.DataFrame(index=sp.index, columns=sp.columns, dtype=float)
    fl_est = pd.DataFrame(index=fl.index, columns=fl.columns, dtype=float)
    for link, nbrs in neighbors.items():
        cols = [n for n in nbrs if n in sp.columns]
        if link in sp.columns and cols:
            sp_est[link] = sp[cols].mean(axis=1)
            fl_est[link] = fl[cols].mean(axis=1)
    m = m.merge(sp_est.stack().rename("sp_speed").reset_index(), on=["timestamp", "link_id"], how="left")
    m = m.merge(fl_est.stack().rename("sp_flow").reset_index(), on=["timestamp", "link_id"], how="left")

    # join truth
    m = m.merge(t.rename(columns={"speed_kmh": "true_speed", "flow_vph": "true_flow"}),
                on=["timestamp", "link_id"], how="left")
    masked = m.speed_kmh.isna() & m.true_speed.notna()
    d = m[masked].dropna(subset=["ti_speed", "sp_speed", "ti_flow", "sp_flow"])
    rows.append(d[["true_speed", "true_flow", "ti_speed", "ti_flow", "sp_speed", "sp_flow"]])

D = pd.concat(rows, ignore_index=True)
print(f"\nmasked cells with both estimates: {len(D):,}")

for var in ["speed", "flow"]:
    y = D[f"true_{var}"].to_numpy()
    ti = D[f"ti_{var}"].to_numpy()
    sp = D[f"sp_{var}"].to_numpy()
    rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
    print(f"\n[{var}]  n={len(y):,}")
    print(f"  temporal-only RMSE: {rmse(ti):.3f}")
    print(f"  spatial-only  RMSE: {rmse(sp):.3f}")
    # least-squares blend: y ~ a*ti + b*sp + c  (fit on masked, report in-sample)
    A = np.column_stack([ti, sp, np.ones_like(y)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    blend = A @ coef
    print(f"  LS blend RMSE:      {rmse(blend):.3f}   (coef ti={coef[0]:.3f}, sp={coef[1]:.3f}, bias={coef[2]:.3f})")
    # correlation between ti and sp residuals
    rt = ti - y
    rs = sp - y
    corr = float(np.corrcoef(rt, rs)[0, 1])
    print(f"  residual corr(temporal, spatial) = {corr:.3f}")
