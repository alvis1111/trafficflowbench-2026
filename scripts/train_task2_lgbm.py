"""Train Task 2 queue LightGBM (vectorized). History features -> future queue."""
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"
SCORED_PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W",
                 "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]
FEAT_COLS = ["v_last_over_vf", "v_min15_over_vf", "dv_dt", "d2v",
             "q_last_over_cap", "occ_last", "occ_trend", "is_queued_T",
             "dt_min", "hour", "weekday"]
DELTAS = [5, 10, 15, 20, 25, 30]  # forecast horizons (min)


def load_fd(panel):
    fd = pd.read_csv(f"{ROOT}/corridors/{panel}/network/fd_parameters.csv", dtype={"link_id": str})
    fd = fd.groupby("link_id", as_index=False)[["free_speed_kmh", "capacity_vph"]].mean()
    links = pd.read_csv(f"{ROOT}/corridors/{panel}/network/links.csv", dtype={"link_id": str})
    v_cut = links.set_index("link_id").free_speed_kmh * 0.6
    fd = fd.merge(v_cut.rename("v_cut").reset_index(), on="link_id", how="left")
    return fd.set_index("link_id")


def day_features(panel, day_file, fd):
    d = pd.read_parquet(day_file, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy"])
    d["link_id"] = d.link_id.astype(str); d["ts"] = pd.to_datetime(d.timestamp, utc=True)
    d = d.sort_values(["link_id", "ts"])
    g = d.groupby("link_id", sort=False)
    sp = g.speed_kmh
    fl = g.flow_vph
    occ = g.occupancy
    vf = d.link_id.map(fd.free_speed_kmh).to_numpy()
    cap = d.link_id.map(fd.capacity_vph).to_numpy()
    vc = d.link_id.map(fd.v_cut).to_numpy()
    v = pd.to_numeric(d.speed_kmh, errors="coerce").to_numpy()
    f = pd.to_numeric(d.flow_vph, errors="coerce").to_numpy()
    o = pd.to_numeric(d.occupancy, errors="coerce").to_numpy()
    v_last = sp.transform("last")
    v_min15 = sp.transform(lambda s: s.rolling(3, min_periods=1).min())
    v_12ago = sp.shift(12)
    dv = (v_last - v_12ago) / 12.0 / 5.0  # per min
    v_1ago = sp.shift(1); v_2ago = sp.shift(2)
    d2v = v_last - 2 * v_1ago + v_2ago
    q_last = fl.transform("last")
    occ_last = occ.transform("last")
    occ_12ago = occ.shift(12)
    occ_trend = occ_last - occ_12ago
    is_q = (v <= vc).astype(int)
    hour = d.ts.dt.hour.to_numpy(); wd = d.ts.dt.weekday.to_numpy()
    # base features (per cell at time T)
    base = np.column_stack([
        np.where(np.isfinite(v), v, 0) / np.maximum(vf, 1),
        np.where(np.isfinite(v_min15.to_numpy()), v_min15.to_numpy(), 0) / np.maximum(vf, 1),
        np.where(np.isfinite(dv.to_numpy()), dv.to_numpy(), 0),
        np.where(np.isfinite(d2v.to_numpy()), d2v.to_numpy(), 0),
        np.where(np.isfinite(f), f, 0) / np.maximum(cap, 1),
        np.where(np.isfinite(o), o, 0),
        np.where(np.isfinite(occ_trend.to_numpy()), occ_trend.to_numpy(), 0),
        is_q,
    ])
    # label: queue at t+delta, for each delta
    q_now = (v <= vc).astype(int)
    rows = []
    for delta in DELTAS:
        steps = delta // 5
        q_future = pd.Series(q_now).groupby(d.link_id).shift(-steps).to_numpy()
        lab = q_future
        feats = np.column_stack([base, np.full(len(d), delta), hour, wd])
        valid = np.isfinite(lab)
        rows.append((feats[valid], lab[valid]))
    return rows


def main():
    frames = []
    for p in SCORED_PANELS:
        fd = load_fd(p)
        days = sorted(glob.glob(f"{ROOT}/corridors/{p}/train/mainline_states/year_month=*/*.parquet"))[:20]
        for day in days:
            for feats, lab in day_features(p, day, fd):
                frames.append((feats, lab))
    X = np.vstack([f for f, _ in frames]); y = np.concatenate([l for _, l in frames])
    print(f"training rows: {len(y):,}  positive rate: {y.mean():.4f}", flush=True)

    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                               class_weight="balanced", verbose=-1)
    model.fit(X, y)
    model.booster_.save_model("task2_lgbm.txt")
    print("saved task2_lgbm.txt", flush=True)


if __name__ == "__main__":
    main()
