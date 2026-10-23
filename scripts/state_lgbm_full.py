"""Full LightGBM state pipeline: train on all 10 panels, predict val+private, build submission."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent / "repo" / "src"))
from task1.baseline_task1_historical_mean import build_profile, slot_values

PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S",
          "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]
FEATS = ["interp_speed", "hist_speed", "spatial_speed", "interp_flow", "hist_flow", "spatial_flow",
         "lanes", "free_speed", "capacity", "length", "weekday", "tod"]
OUT_COLS = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]


def neighbor_map(panel_dir):
    topo = pd.read_csv(panel_dir / "network" / "lwr_mainline_topology.csv", dtype=str)
    def main(s):
        return [x for x in str(s).split(";") if x and x.startswith("L")] if pd.notna(s) else []
    return {r["mainline_link_id"]: main(r.get("incoming_link_ids")) + main(r.get("outgoing_link_ids"))
            for _, r in topo.iterrows()}


def _interp(s):
    return s.interpolate(method="linear", limit_direction="both")


def build_day(m, sp, fp, li_map, nb, static):
    """Compute features for one masked day (DataFrame m, already truth-merged if train)."""
    m = m.sort_values(["link_id", "timestamp"]).reset_index(drop=True)
    interp_speed = m.groupby("link_id")["speed_kmh"].transform(_interp).to_numpy(dtype=float)
    wd, tod = slot_values(m)
    slot = wd * 288 + tod
    li = m.link_id.map(li_map).fillna(-1).to_numpy(dtype=np.int64)
    safe = np.where(li >= 0, li, 0)
    base_speed = sp["mean"][safe, slot]
    base_flow = fp["mean"][safe, slot]
    m["_rf"] = m["flow_vph"].to_numpy(dtype=float) - base_flow
    interp_flow = base_flow + m.groupby("link_id")["_rf"].transform(_interp).to_numpy(dtype=float)

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
        sp_sp.stack().rename("ss").reset_index(), on=["timestamp", "link_id"], how="left")["ss"].to_numpy()
    spatial_flow = m[["timestamp", "link_id"]].merge(
        sp_fl.stack().rename("sf").reset_index(), on=["timestamp", "link_id"], how="left")["sf"].to_numpy()
    spatial_speed = np.where(np.isfinite(spatial_speed), spatial_speed, interp_speed)
    spatial_flow = np.where(np.isfinite(spatial_flow), spatial_flow, interp_flow)

    out = pd.DataFrame({
        "interp_speed": interp_speed, "hist_speed": base_speed, "spatial_speed": spatial_speed,
        "interp_flow": interp_flow, "hist_flow": base_flow, "spatial_flow": spatial_flow,
        "link_id": m.link_id.to_numpy(), "weekday": wd, "tod": tod,
        "timestamp": m.timestamp.to_numpy(), "station_id": m.station_id.to_numpy(),
        "mask_regime": m.mask_regime.to_numpy(), "date": m.date.to_numpy(),
    })
    out = out.join(static, on="link_id")
    out["masked"] = m.speed_kmh.isna().to_numpy()
    if "true_speed" in m.columns:
        out["true_speed"] = m.true_speed.to_numpy()
        out["true_flow"] = m.true_flow.to_numpy()
    return out


def process_panel(panel, release, split, need_truth):
    panel_dir = release / "corridors" / panel
    nb = neighbor_map(panel_dir)
    links = pd.read_csv(panel_dir / "network" / "links.csv", dtype={"link_id": str})
    static = links.set_index("link_id")[["lanes", "free_speed_kmh", "capacity_vph", "length_km"]]
    static = static.rename(columns={"free_speed_kmh": "free_speed", "capacity_vph": "capacity", "length_km": "length"})
    sp, fp, _ = build_profile(panel, panel_dir)
    li_map = sp["link_index"]

    cols = ["date", "timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "mask_regime"]
    masked_files = sorted((panel_dir / split / "mainline_states_masked").glob("**/*.parquet"))
    truth_files = {p.name: p for p in (panel_dir / split / "mainline_states").glob("**/*.parquet")} if need_truth else {}
    chunks = []
    for mf in masked_files:
        m = pd.read_parquet(mf, columns=cols)
        m["link_id"] = m.link_id.astype(str); m["station_id"] = m.station_id.astype(str)
        if need_truth:
            t = pd.read_parquet(truth_files[mf.name],
                                columns=["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph"])
            t["link_id"] = t.link_id.astype(str); t["station_id"] = t.station_id.astype(str)
            m = m.merge(t.rename(columns={"speed_kmh": "true_speed", "flow_vph": "true_flow"}),
                        on=["timestamp", "station_id", "link_id"], how="left")
        chunks.append(build_day(m, sp, fp, li_map, nb, static))
    return pd.concat(chunks, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", type=Path, default=Path("C:/Users/GEM07/OneDrive/Desktop/kaggle_public"))
    ap.add_argument("--output-dir", type=Path, default=Path("C:/Users/GEM07/kaggle-traffic-flow"))
    args = ap.parse_args()
    release = args.release_root.resolve()

    # --- train ---
    print("extracting train features ...", flush=True)
    tr_chunks = []
    for p in PANELS:
        print(f"  train {p}", flush=True)
        tr_chunks.append(process_panel(p, release, "train", True)[FEATS + ["true_speed", "true_flow", "masked"]])
    tr = pd.concat(tr_chunks, ignore_index=True)
    tr = tr[tr.masked & tr.true_speed.notna() & tr.true_flow.notna()]
    print(f"train rows: {len(tr):,}", flush=True)

    speed_model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                    subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
    flow_model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                   subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
    print("training speed model ...", flush=True)
    speed_model.fit(tr[FEATS], tr.true_speed.to_numpy())
    print("training flow model ...", flush=True)
    flow_model.fit(tr[FEATS], tr.true_flow.to_numpy())

    # --- predict val + private ---
    for split in ["validation", "private"]:
        print(f"predicting {split} ...", flush=True)
        outs = []
        for p in PANELS:
            f = process_panel(p, release, split, False)
            # template targets
            tmpl = pd.read_csv(release / "task1" / p / split / "sample_submission_state.csv",
                               usecols=["timestamp", "station_id", "link_id"], dtype=str)
            tmpl["_k"] = tmpl.timestamp + "|" + tmpl.station_id + "|" + tmpl.link_id
            f["_k"] = f.timestamp.astype(str) + "|" + f.station_id.astype(str) + "|" + f.link_id.astype(str)
            f = f[f._k.isin(set(tmpl._k))].copy()
            f["speed_kmh"] = np.maximum(speed_model.predict(f[FEATS]), 0.0)
            f["flow_vph"] = np.maximum(flow_model.predict(f[FEATS]), 0.0)
            f["panel"] = p
            outs.append(f[OUT_COLS])
        out = pd.concat(outs, ignore_index=True)
        out.to_csv(args.output_dir / f"lgbm_state_{split}.csv", index=False)
        print(f"  wrote {len(out):,} rows to lgbm_state_{split}.csv", flush=True)


if __name__ == "__main__":
    main()
