"""CTM Stage 0: reverse-engineer the generator.

Check whether the observed data obeys (a) the triangular fundamental diagram
from fd_parameters.csv and (b) link conservation.  This tells us if standard
CTM is the right model before building anything.
"""
import glob
import os
import numpy as np
import pandas as pd

PANEL = "D12_I405_S"
ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"
PD = f"{ROOT}/corridors/{PANEL}"

# --- network ---
fd = pd.read_csv(f"{PD}/network/fd_parameters.csv", dtype={"link_id": str})
links = pd.read_csv(f"{PD}/network/links.csv", dtype={"link_id": str})
topo = pd.read_csv(f"{PD}/network/lwr_mainline_topology.csv", dtype=str)

# per-link FD params (per-lane)
fd = fd.groupby("link_id", as_index=False).agg(
    free_speed=("free_speed_kmh", "mean"), cap=("capacity_vph", "mean"),
    kcrit=("critical_density", "mean"), kjam=("k_jam", "mean"), lanes=("lanes", "mean"))
fd["cap_lane"] = fd.cap / fd.lanes
fd["kcrit_lane"] = fd.kcrit / fd.lanes
fd["kjam_lane"] = fd.kjam / fd.lanes
params = fd.set_index("link_id")

# incoming/outgoing map (mainline only)
def main_ids(s):
    return [x for x in str(s).split(";") if x and x.startswith("L")] if pd.notna(s) else []
inc = {r["mainline_link_id"]: main_ids(r.get("incoming_link_ids")) for _, r in topo.iterrows()}
out = {r["mainline_link_id"]: main_ids(r.get("outgoing_link_ids")) for _, r in topo.iterrows()}

# --- one day ---
day_file = sorted(glob.glob(f"{PD}/train/mainline_states/year_month=*/*.parquet"))[0]
d = pd.read_parquet(day_file, columns=["timestamp", "link_id", "speed_kmh", "flow_vph"])
d["link_id"] = d.link_id.astype(str); d["ts"] = pd.to_datetime(d.timestamp, utc=True)
d = d.merge(params.reset_index(), on="link_id", how="inner")
d = d.dropna(subset=["speed_kmh", "flow_vph"])
d["k"] = d.flow_vph / d.speed_kmh.clip(lower=1.0)  # total density
d["k_lane"] = d.k / d.lanes
d["q_lane"] = d.flow_vph / d.lanes

# --- (a) FD check: q_lane vs triangular FD at k_lane ---
d["q_fd"] = np.where(d.k_lane <= d.kcrit_lane,
                     d.free_speed * d.k_lane,
                     (d.cap_lane / (d.kjam_lane - d.kcrit_lane)) * np.maximum(d.kjam_lane - d.k_lane, 0))
d["fd_err"] = (d.q_lane - d.q_fd).abs() / d.q_lane.clip(lower=1.0)
print("=== (a) 基本图一致性 ===")
print(f"观测样本: {len(d):,}")
print(f"FD 相对误差中位数: {d.fd_err.median():.4f}")
print(f"FD 相对误差 p90: {d.fd_err.quantile(0.9):.4f}")
print(f"自由流占比(k<=kcrit): {(d.k_lane <= d.kcrit_lane).mean():.3f}")
# how well does q_lane match q_fd?
r = 1 - d.fd_err.sum() / len(d)
print(f"近似 S_FD (1 - mean|err|): {r:.4f}")

# --- (b) conservation check on observed data ---
print("\n=== (b) 守恒律检验 ===")
# pivot flow per (ts, link)
pivot_q = d.pivot_table(index="ts", columns="link_id", values="flow_vph")
pivot_v = d.pivot_table(index="ts", columns="link_id", values="speed_kmh")
lengths = links.set_index("link_id")["length_km"]
lanes = links.set_index("link_id")["lanes"]
resids = []
for link in pivot_q.columns:
    if link not in inc or link not in out:
        continue
    L = lengths.get(link, 1.0)
    q = pivot_q[link]
    v = pivot_v[link].clip(lower=1.0)
    N = (q / v) * L  # accumulation
    dN = N.diff().shift(-1)  # N(t+dt) - N(t)
    # inflow/outflow from mainline neighbours
    qi = pd.Series(0.0, index=pivot_q.index)
    qo = pd.Series(0.0, index=pivot_q.index)
    for c in inc.get(link, []):
        if c in pivot_q.columns:
            qi = qi + pivot_q[c]
    for c in out.get(link, []):
        if c in pivot_q.columns:
            qo = qo + pivot_q[c]
    rhs = (qi - qo) * (5.0 / 60.0)  # no ramp for now (mainline-only segments)
    m = dN.notna() & rhs.notna()
    if m.sum() > 10 and rhs.abs().sum() > 1e-9:
        e = ((dN - rhs).abs().sum() / rhs.abs().sum())
        resids.append(e)
resids = np.array(resids)
print(f"参与守恒的 link 数: {len(resids)}")
print(f"守恒残差(mean |dN-rhs|/|rhs|) 中位数: {np.median(resids):.4f}")
print(f"守恒残差 p90: {np.quantile(resids, 0.9):.4f}")
