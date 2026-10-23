"""Train Task 2 LightGBM v2: + mu_over_cap / q_freq historical features, milder class weight."""
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"
SCORED_PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W",
                 "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]
DELTAS = [5, 10, 15, 20, 25, 30]
FEAT_COLS = ["v_last_over_vf", "v_min15_over_vf", "dv_dt", "d2v",
             "q_last_over_cap", "occ_last", "occ_trend", "is_queued_T",
             "mu_over_cap", "q_freq", "dt_min", "hour", "weekday"]


def load_fd(panel):
    fd = pd.read_csv(f"{ROOT}/corridors/{panel}/network/fd_parameters.csv", dtype={"link_id": str})
    fd = fd.groupby("link_id", as_index=False)[["free_speed_kmh", "capacity_vph"]].mean()
    links = pd.read_csv(f"{ROOT}/corridors/{panel}/network/links.csv", dtype={"link_id": str})
    v_cut = links.set_index("link_id").free_speed_kmh * 0.6
    fd = fd.merge(v_cut.rename("v_cut").reset_index(), on="link_id", how="left")
    return fd.set_index("link_id")


def build_hist_profiles(panel, fd):
    """mu_over_cap and q_freq per (link, dow, tod)."""
    link_ids = sorted(fd.index.tolist())
    li = {l: i for i, l in enumerate(link_ids)}
    n = len(link_ids)
    mu_sum = np.zeros((n, 7, 288)); mu_cnt = np.zeros((n, 7, 288))
    qf_sum = np.zeros((n, 7, 288)); qf_cnt = np.zeros((n, 7, 288))
    for day in sorted(glob.glob(f"{ROOT}/corridors/{panel}/train/mainline_states/year_month=*/*.parquet")):
        d = pd.read_parquet(day, columns=["timestamp", "link_id", "speed_kmh", "flow_vph"])
        d["link_id"] = d.link_id.astype(str); d["ts"] = pd.to_datetime(d.timestamp, utc=True)
        d = d[d.link_id.isin(link_ids)]
        dow = d.ts.dt.weekday.to_numpy(); tod = (d.ts.dt.hour * 12 + d.ts.dt.minute // 5).to_numpy()
        idx = d.link_id.map(li).to_numpy()
        cap = d.link_id.map(fd.capacity_vph).to_numpy(); vc = d.link_id.map(fd.v_cut).to_numpy()
        f = pd.to_numeric(d.flow_vph, errors="coerce").to_numpy()
        s = pd.to_numeric(d.speed_kmh, errors="coerce").to_numpy()
        mu = f / np.maximum(cap, 1)
        qf = (s <= vc).astype(float)
        valid = np.isfinite(mu) & np.isfinite(qf)
        np.add.at(mu_sum, (idx[valid], dow[valid], tod[valid]), mu[valid])
        np.add.at(mu_cnt, (idx[valid], dow[valid], tod[valid]), 1.0)
        np.add.at(qf_sum, (idx[valid], dow[valid], tod[valid]), qf[valid])
        np.add.at(qf_cnt, (idx[valid], dow[valid], tod[valid]), 1.0)
    mu = mu_sum / np.maximum(mu_cnt, 1)
    qf = qf_sum / np.maximum(qf_cnt, 1)
    return link_ids, li, mu, qf


def main():
    X_list, y_list = [], []
    for p in SCORED_PANELS:
        fd = load_fd(p)
        link_ids, li, mu, qf = build_hist_profiles(p, fd)
        print(f"{p}: {len(link_ids)} links", flush=True)
        days = sorted(glob.glob(f"{ROOT}/corridors/{p}/train/mainline_states/year_month=*/*.parquet"))[:20]
        for day in days:
            d = pd.read_parquet(day, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy"])
            d["link_id"] = d.link_id.astype(str); d["ts"] = pd.to_datetime(d.timestamp, utc=True)
            d = d[d.link_id.isin(link_ids)].sort_values(["link_id", "ts"])
            g = d.groupby("link_id", sort=False)
            vf = d.link_id.map(fd.free_speed_kmh).to_numpy(); cap = d.link_id.map(fd.capacity_vph).to_numpy()
            vc = d.link_id.map(fd.v_cut).to_numpy()
            v = pd.to_numeric(d.speed_kmh, errors="coerce").to_numpy()
            f = pd.to_numeric(d.flow_vph, errors="coerce").to_numpy()
            o = pd.to_numeric(d.occupancy, errors="coerce").to_numpy()
            dow = d.ts.dt.weekday.to_numpy(); tod = (d.ts.dt.hour * 12 + d.ts.dt.minute // 5).to_numpy()
            idx = d.link_id.map(li).to_numpy()
            sp = g.speed_kmh
            v_last = sp.transform("last").to_numpy()
            v_min15 = sp.transform(lambda s: s.rolling(3, min_periods=1).min()).to_numpy()
            dv = (v_last - sp.shift(12).to_numpy()) / 12.0 / 5.0
            d2v = v_last - 2 * sp.shift(1).to_numpy() + sp.shift(2).to_numpy()
            q_last = g.flow_vph.transform("last").to_numpy()
            occ_last = g.occupancy.transform("last").to_numpy()
            occ_trend = occ_last - g.occupancy.shift(12).to_numpy()
            is_q = (v <= vc).astype(int)
            base = np.column_stack([
                np.nan_to_num(v) / np.maximum(vf, 1), np.nan_to_num(v_min15) / np.maximum(vf, 1),
                np.nan_to_num(dv), np.nan_to_num(d2v),
                np.nan_to_num(f) / np.maximum(cap, 1), np.nan_to_num(o), np.nan_to_num(occ_trend),
                is_q,
                mu[idx, dow, tod], qf[idx, dow, tod],
            ])
            q_now = (v <= vc).astype(int)
            for delta in DELTAS:
                steps = delta // 5
                q_fut = pd.Series(q_now).groupby(d.link_id).shift(-steps).to_numpy()
                lab = q_fut
                feats = np.column_stack([base, np.full(len(d), delta), dow, np.full(len(d), 0)])
                # hour needs per-cell; recompute
                feats[:, -2] = d.ts.dt.hour.to_numpy()
                feats[:, -1] = dow
                valid = np.isfinite(lab)
                X_list.append(feats[valid]); y_list.append(lab[valid])
    X = np.vstack(X_list); y = np.concatenate(y_list)
    print(f"training rows: {len(y):,} positive rate: {y.mean():.4f}", flush=True)
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                               class_weight={0: 1, 1: 15}, verbose=-1)
    model.fit(X, y)
    model.booster_.save_model("task2_lgbm_v2.txt")
    print("saved task2_lgbm_v2.txt", flush=True)


if __name__ == "__main__":
    main()
