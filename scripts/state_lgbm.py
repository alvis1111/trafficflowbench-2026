"""Task 1 state reconstruction with LightGBM: spatial + temporal + prior features.

Features per masked cell (link L, time T):
  - interp_speed / interp_flow  : same-day temporal interpolation (v2 method)
  - hist_speed  / hist_flow     : weekday x time-of-day historical prior
  - spatial_speed / spatial_flow: mean of up/downstream neighbours' observed value at T
  - lanes, free_speed, capacity, length : static link features
  - weekday, tod                 : time features

Trains separate LightGBM regressors for speed and flow, evaluated out-of-sample
on held-out days.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent / "repo" / "src"))
from task1.baseline_task1_historical_mean import build_profile, slot_values


def neighbor_map(panel_dir: Path) -> dict:
    topo = pd.read_csv(panel_dir / "network" / "lwr_mainline_topology.csv", dtype=str)
    def main(s):
        return [x for x in str(s).split(";") if x and x.startswith("L")] if pd.notna(s) else []
    return {r["mainline_link_id"]: main(r.get("incoming_link_ids")) + main(r.get("outgoing_link_ids"))
            for _, r in topo.iterrows()}


def _interp(s):
    return s.interpolate(method="linear", limit_direction="both")


def build_features(panel: str, release: Path, split: str) -> pd.DataFrame:
    panel_dir = release / "corridors" / panel
    nb = neighbor_map(panel_dir)
    links = pd.read_csv(panel_dir / "network" / "links.csv", dtype={"link_id": str})
    static = links.set_index("link_id")[["lanes", "free_speed_kmh", "capacity_vph", "length_km"]]
    static = static.rename(columns={"free_speed_kmh": "free_speed", "capacity_vph": "capacity", "length_km": "length"})

    sp, fp, _ = build_profile(panel, panel_dir)
    li_map = sp["link_index"]

    masked_files = sorted((panel_dir / split / "mainline_states_masked").glob("**/*.parquet"))
    truth_files = {p.name: p for p in (panel_dir / split / "mainline_states").glob("**/*.parquet")}

    chunks = []
    for mf in masked_files:
        tf = truth_files.get(mf.name)
        if tf is None:
            continue
        m = pd.read_parquet(mf, columns=["date", "timestamp", "station_id", "link_id", "speed_kmh", "flow_vph"])
        t = pd.read_parquet(tf, columns=["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph"])
        m["link_id"] = m.link_id.astype(str); m["station_id"] = m.station_id.astype(str)
        t["link_id"] = t.link_id.astype(str); t["station_id"] = t.station_id.astype(str)
        m = m.merge(t.rename(columns={"speed_kmh": "true_speed", "flow_vph": "true_flow"}),
                    on=["timestamp", "station_id", "link_id"], how="left")
        m = m.sort_values(["link_id", "timestamp"]).reset_index(drop=True)

        # temporal interpolation (v2): speed raw, flow residual
        interp_speed = m.groupby("link_id")["speed_kmh"].transform(_interp).to_numpy(dtype=float)
        wd, tod = slot_values(m)
        slot = wd * 288 + tod
        li = m.link_id.map(li_map).fillna(-1).to_numpy(dtype=np.int64)
        safe = np.where(li >= 0, li, 0)
        base_speed = sp["mean"][safe, slot]
        base_flow = fp["mean"][safe, slot]
        m["_rf"] = m["flow_vph"].to_numpy(dtype=float) - base_flow
        interp_resid_flow = m.groupby("link_id")["_rf"].transform(_interp).to_numpy(dtype=float)
        interp_flow = base_flow + interp_resid_flow

        # spatial: neighbour observed mean at same time
        obs_sp = m.pivot_table(index="timestamp", columns="link_id", values="speed_kmh")
        obs_fl = m.pivot_table(index="timestamp", columns="link_id", values="flow_vph")
        sp_sp = pd.DataFrame(index=obs_sp.index, columns=obs_sp.columns, dtype=float)
        sp_fl = pd.DataFrame(index=obs_fl.index, columns=obs_fl.columns, dtype=float)
        for link, nbrs in nb.items():
            cols = [n for n in nbrs if n in obs_sp.columns]
            if link in obs_sp.columns and cols:
                sp_sp[link] = obs_sp[cols].mean(axis=1)
                sp_fl[link] = obs_fl[cols].mean(axis=1)
        spatial_speed = m[["timestamp", "link_id"]].merge(
            sp_sp.stack().rename("spatial_speed").reset_index(), on=["timestamp", "link_id"], how="left")["spatial_speed"].to_numpy()
        spatial_flow = m[["timestamp", "link_id"]].merge(
            sp_fl.stack().rename("spatial_flow").reset_index(), on=["timestamp", "link_id"], how="left")["spatial_flow"].to_numpy()
        # fall back to the temporal interpolation where spatial has no observed neighbours
        spatial_speed = np.where(np.isfinite(spatial_speed), spatial_speed, interp_speed)
        spatial_flow = np.where(np.isfinite(spatial_flow), spatial_flow, interp_flow)

        d = pd.DataFrame({
            "interp_speed": interp_speed, "hist_speed": base_speed, "spatial_speed": spatial_speed,
            "interp_flow": interp_flow, "hist_flow": base_flow, "spatial_flow": spatial_flow,
            "link_id": m.link_id.to_numpy(), "weekday": wd, "tod": tod,
            "date": m.date.to_numpy(),
            "true_speed": m.true_speed.to_numpy(), "true_flow": m.true_flow.to_numpy(),
        })
        d = d.join(static, on="link_id")
        masked = m.speed_kmh.isna().to_numpy() & m.true_speed.notna().to_numpy() & m.true_flow.notna().to_numpy()
        chunks.append(d[masked])

    return pd.concat(chunks, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", type=Path, default=Path("C:/Users/GEM07/OneDrive/Desktop/kaggle_public"))
    ap.add_argument("--panel", default="D12_I405_S")
    args = ap.parse_args()

    print(f"extracting features for {args.panel} ...", flush=True)
    df = build_features(args.panel, args.release_root.resolve(), "train")
    print(f"features: {len(df):,} rows", flush=True)

    # clean out-of-sample split by DAY: hold out the last 15% of unique dates
    dates = sorted(df.date.unique())
    n_dates = len(dates)
    val_dates = set(dates[int(n_dates * 0.85):])
    tr = df[~df.date.isin(val_dates)]
    va = df[df.date.isin(val_dates)]
    print(f"train {len(tr):,} rows ({len(dates)-len(val_dates)} days) / val {len(va):,} rows ({len(val_dates)} days)", flush=True)

    for target, cols in [("speed", ["interp_speed", "hist_speed", "spatial_speed"]),
                         ("flow", ["interp_flow", "hist_flow", "spatial_flow"])]:
        feats = cols + ["lanes", "free_speed", "capacity", "length", "weekday", "tod"]
        y = tr[f"true_{target}"].to_numpy()
        model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                  subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
        model.fit(tr[feats], y)
        pred = model.predict(va[feats])
        base = va[cols[0]].to_numpy()  # interp_* is the v2 prediction
        yv = va[f"true_{target}"].to_numpy()
        rmse_lgb = float(np.sqrt(np.mean((pred - yv) ** 2)))
        rmse_v2 = float(np.sqrt(np.mean((base - yv) ** 2)))
        print(f"[{target}] v2 RMSE={rmse_v2:.3f}  lightgbm RMSE={rmse_lgb:.3f}  ({(1 - rmse_lgb / rmse_v2) * 100:+.1f}%)", flush=True)


if __name__ == "__main__":
    main()
