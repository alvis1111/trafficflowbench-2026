"""Evaluate the v2 LightGBM hybrid on train windows, with threshold sweep."""
import glob
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"
SCORED_PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W",
                 "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]
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


def get_onset_cluster(panel, T):
    hr = int(T.hour)
    if panel == "D12_I5_N": return ["L5N-147", "L5N-235", "L5N-237"] if hr >= 13 else ["L5N-059", "L5N-061", "L5N-104"]
    elif panel == "D12_I5_S": return ["L5S-152", "L5S-249", "L5S-065"] if hr >= 11 else ["L5S-051", "L5S-101", "L5S-132", "L5S-278"]
    elif panel == "D7_I10_E": return ["L10E-043", "L10E-052", "L10E-053", "L10E-124", "L10E-125", "L10E-180", "L10E-243"]
    elif panel == "D7_I10_W": return ["L10W-212", "L10W-013"]
    elif panel == "D7_I210_E": return ["L210E-001", "L210E-129", "L210E-131", "L210E-132", "L210E-261"]
    elif panel == "D7_I210_W": return ["L210W-190", "L210W-246"] if hr in (8, 9) else ["L210W-004", "L210W-028", "L210W-029", "L210W-172"]
    elif panel == "D7_I405_N": return ["L405N-041", "L405N-145"]
    elif panel == "D7_I405_S": return ["L405S-127", "L405S-317", "L405S-319", "L405S-373"] if hr in (9, 10, 11, 12) else ["L405S-001", "L405S-017", "L405S-056", "L405S-065", "L405S-260", "L405S-264"]
    return []


def link_features(h_df, fd):
    feats = {}
    for lid, h in h_df.groupby("link_id"):
        sp = pd.to_numeric(h.speed_kmh, errors="coerce").to_numpy(); sp = sp[np.isfinite(sp)]
        if len(sp) < 1: continue
        vf = fd.loc[str(lid), "free_speed_kmh"]; cap = fd.loc[str(lid), "capacity_vph"]; vc = fd.loc[str(lid), "v_cut"]
        fl = pd.to_numeric(h.flow_vph, errors="coerce").to_numpy(); fl = fl[np.isfinite(fl)]
        occ = pd.to_numeric(h.occupancy, errors="coerce").to_numpy(); occ = occ[np.isfinite(occ)]
        v_last = sp[-1]; v_min15 = sp[-3:].min()
        dv = (sp[-1] - sp[0]) / max(len(sp) - 1, 1) / 5.0
        d2v = sp[-1] - 2 * sp[-2] + sp[-3] if len(sp) >= 3 else 0.0
        q_ratio = fl[-1] / cap if len(fl) else 0.0
        occ_last = occ[-1] if len(occ) else 0.0
        occ_trend = (occ[-1] - occ[0]) / max(len(occ) - 1, 1) if len(occ) > 1 else 0.0
        feats[str(lid)] = [v_last / max(vf, 1), v_min15 / max(vf, 1), dv, d2v,
                           q_ratio, occ_last, occ_trend, int(v_last <= vc)]
    return feats


def build_hist(panel, fd):
    link_ids = sorted(fd.index.tolist()); li = {l: i for i, l in enumerate(link_ids)}
    n = len(link_ids)
    mu_sum = np.zeros((n, 7, 288)); mu_cnt = np.zeros((n, 7, 288))
    qf_sum = np.zeros((n, 7, 288)); qf_cnt = np.zeros((n, 7, 288))
    for day in glob.glob(f"{ROOT}/corridors/{panel}/train/mainline_states/year_month=*/*.parquet"):
        d = pd.read_parquet(day, columns=["timestamp", "link_id", "speed_kmh", "flow_vph"])
        d["link_id"] = d.link_id.astype(str); d["ts"] = pd.to_datetime(d.timestamp, utc=True)
        d = d[d.link_id.isin(link_ids)]
        dow = d.ts.dt.weekday.to_numpy(); tod = (d.ts.dt.hour * 12 + d.ts.dt.minute // 5).to_numpy()
        idx = d.link_id.map(li).to_numpy()
        cap = d.link_id.map(fd.capacity_vph).to_numpy(); vc = d.link_id.map(fd.v_cut).to_numpy()
        mu = pd.to_numeric(d.flow_vph, errors="coerce").to_numpy() / np.maximum(cap, 1)
        qf = (pd.to_numeric(d.speed_kmh, errors="coerce").to_numpy() <= vc).astype(float)
        valid = np.isfinite(mu) & np.isfinite(qf)
        np.add.at(mu_sum, (idx[valid], dow[valid], tod[valid]), mu[valid])
        np.add.at(mu_cnt, (idx[valid], dow[valid], tod[valid]), 1.0)
        np.add.at(qf_sum, (idx[valid], dow[valid], tod[valid]), qf[valid])
        np.add.at(qf_cnt, (idx[valid], dow[valid], tod[valid]), 1.0)
    return li, mu_sum / np.maximum(mu_cnt, 1), qf_sum / np.maximum(qf_cnt, 1)


def evaluate(th_on, th_off):
    model = lgb.Booster(model_file="task2_lgbm_v2.txt")
    rows = []
    for panel in SCORED_PANELS:
        fd = load_fd(panel)
        li, mu, qf = build_hist(panel, fd)
        widx = pd.read_csv(f"{ROOT}/task2/{panel}/train/window_index.csv")
        hist = pd.read_parquet(f"{ROOT}/task2/{panel}/train/window_history.parquet")
        tmpl = pd.read_csv(f"{ROOT}/task2/{panel}/train/sample_submission_queue.csv")
        links = pd.read_csv(f"{ROOT}/corridors/{panel}/network/links.csv", dtype={"link_id": str})
        v_cut = dict(zip(links.link_id, links.free_speed_kmh * 0.6))
        hist["ts"] = pd.to_datetime(hist.timestamp, utc=True); tmpl["ts"] = pd.to_datetime(tmpl.timestamp, utc=True)
        hist["link_id"] = hist.link_id.astype(str); tmpl["link_id"] = tmpl.link_id.astype(str)
        truth_files = {os.path.basename(p).replace("synthetic_mainline_", "").replace(".parquet", ""): p
                       for p in glob.glob(f"{ROOT}/corridors/{panel}/train/mainline_states/year_month=*/*.parquet")}
        for wid, grp in tmpl.groupby("window_id"):
            w = widx[widx.window_id == wid].iloc[0]
            cond = w.condition
            h = hist[hist.window_id == wid]
            last = h[h.ts == h.ts.max()].drop_duplicates("link_id")
            last["q_now"] = (last.speed_kmh <= last.link_id.map(v_cut)).astype(int)
            q_now = dict(zip(last.link_id, last.q_now))
            cluster = set(get_onset_cluster(panel, pd.Timestamp(w.forecast_origin)))
            t_end = grp.ts.max()
            lf = link_features(h, fd)
            hr = pd.Timestamp(w.forecast_origin).hour; wd = pd.Timestamp(w.forecast_origin).weekday()
            tod_origin = hr * 12 + pd.Timestamp(w.forecast_origin).minute // 5
            preds = []
            for _, r in grp.iterrows():
                lid = str(r.link_id); f = lf.get(lid)
                dt_min = 5.0 + (r.ts - grp.ts.min()).total_seconds() / 60.0
                tod_cell = r.ts.hour * 12 + r.ts.minute // 5
                if f is None:
                    preds.append(int(q_now.get(lid, 0)) if "ongoing" in cond else 0)
                    continue
                mu_f = mu[li[lid], wd, tod_cell]; qf_f = qf[li[lid], wd, tod_cell]
                feat = f + [mu_f, qf_f, dt_min, hr, wd]
                p = model.predict(pd.DataFrame([feat], columns=FEAT_COLS))[0]
                if "ongoing" in cond:
                    base = int(q_now.get(lid, 0))
                    preds.append(1 if base else (1 if p >= th_off else 0))
                else:
                    base = int((lid in cluster) and (r.ts == t_end))
                    preds.append(1 if base else (1 if p >= th_on else 0))
            a = np.asarray(preds)
            tf = truth_files.get(w.date.replace("-", "_"))
            day = pd.read_parquet(tf, columns=["timestamp", "link_id", "speed_kmh"])
            day["link_id"] = day.link_id.astype(str); day["ts"] = pd.to_datetime(day.timestamp, utc=True)
            fstart = pd.Timestamp(w.forecast_start); fend = pd.Timestamp(w.forecast_end)
            tr = day[(day.ts >= fstart) & (day.ts <= fend)].copy()
            tr["q"] = (tr.speed_kmh <= tr.link_id.map(v_cut)).astype(int)
            m = grp.merge(tr[["ts", "link_id", "q"]], on=["ts", "link_id"], how="left")
            b = m.q.fillna(0).astype(int).to_numpy()
            inter = int((a & b).sum()); union = int((a | b).sum())
            rows.append({"condition": cond, "iou": 1.0 if union == 0 else inter / union})
    res = pd.DataFrame(rows)
    return res.iou.mean(), res[res.condition == "queue_onset"].iou.mean(), res[res.condition == "queue_ongoing"].iou.mean()


if __name__ == "__main__":
    print("基线: 纯启发式 overall=0.6693 onset=0.7520 ongoing=0.5865")
    for th_on, th_off in [(0.8, 0.7), (0.9, 0.8), (0.95, 0.9), (0.95, 0.85)]:
        ov, on, og = evaluate(th_on, th_off)
        print(f"th_on={th_on} th_off={th_off}: overall={ov:.4f} onset={on:.4f} ongoing={og:.4f}")
