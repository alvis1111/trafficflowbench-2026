"""Task 2 queue: persistence (ongoing) + empirical bottleneck clusters (onset).

Replicates the 0.809 solution's Task 2 heuristic core.
"""
import glob
import os
import numpy as np
import pandas as pd

ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"
SCORED_PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W",
                 "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]

# Hardcoded empirical bottleneck clusters (mined from train, from the 0.809 solution)
ONSET_BOTTLENECK_CLUSTERS = {
    "D12_I5_N": [["L5N-059", "L5N-061", "L5N-104"], ["L5N-147", "L5N-235", "L5N-237"]],
    "D12_I5_S": [["L5S-152", "L5S-249", "L5S-065"], ["L5S-051", "L5S-101", "L5S-132", "L5S-278"]],
    "D7_I10_E": [["L10E-043", "L10E-052", "L10E-053", "L10E-124", "L10E-125", "L10E-180", "L10E-243"]],
    "D7_I10_W": [["L10W-212", "L10W-013"]],
    "D7_I210_E": [["L210E-001", "L210E-129", "L210E-131", "L210E-132", "L210E-261"]],
    "D7_I210_W": [["L210W-190", "L210W-246"], ["L210W-004", "L210W-028", "L210W-029", "L210W-172"]],
    "D7_I405_N": [["L405N-041", "L405N-145"], ["L405N-080", "L405N-082", "L405N-083", "L405N-217", "L405N-233", "L405N-234", "L405N-235", "L405N-392"]],
    "D7_I405_S": [["L405S-001", "L405S-017", "L405S-056", "L405S-065", "L405S-260", "L405S-264"], ["L405S-127", "L405S-317", "L405S-319", "L405S-373"]],
}


def get_onset_cluster(panel, T):
    hr = int(T.hour)
    if panel == "D12_I5_N":
        return ["L5N-147", "L5N-235", "L5N-237"] if hr >= 13 else ["L5N-059", "L5N-061", "L5N-104"]
    elif panel == "D12_I5_S":
        return ["L5S-152", "L5S-249", "L5S-065"] if hr >= 11 else ["L5S-051", "L5S-101", "L5S-132", "L5S-278"]
    elif panel == "D7_I10_E":
        return ["L10E-043", "L10E-052", "L10E-053", "L10E-124", "L10E-125", "L10E-180", "L10E-243"]
    elif panel == "D7_I10_W":
        return ["L10W-212", "L10W-013"]
    elif panel == "D7_I210_E":
        return ["L210E-001", "L210E-129", "L210E-131", "L210E-132", "L210E-261"]
    elif panel == "D7_I210_W":
        return ["L210W-190", "L210W-246"] if hr in (8, 9) else ["L210W-004", "L210W-028", "L210W-029", "L210W-172"]
    elif panel == "D7_I405_N":
        return ["L405N-041", "L405N-145"]
    elif panel == "D7_I405_S":
        return ["L405S-127", "L405S-317", "L405S-319", "L405S-373"] if hr in (9, 10, 11, 12) else ["L405S-001", "L405S-017", "L405S-056", "L405S-065", "L405S-260", "L405S-264"]
    return []


def build_panel_queue(panel, split):
    # window index + history + template
    widx = pd.read_csv(f"{ROOT}/task2/{panel}/{split}/window_index.csv")
    hist = pd.read_parquet(f"{ROOT}/task2/{panel}/{split}/window_history.parquet")
    tmpl = pd.read_csv(f"{ROOT}/task2/{panel}/{split}/sample_submission_queue.csv")
    links = pd.read_csv(f"{ROOT}/corridors/{panel}/network/links.csv", dtype={"link_id": str})
    v_cut = dict(zip(links.link_id, links.free_speed_kmh * 0.6))

    hist["ts"] = pd.to_datetime(hist.timestamp, utc=True)
    tmpl["ts"] = pd.to_datetime(tmpl.timestamp, utc=True)
    hist["link_id"] = hist.link_id.astype(str)
    tmpl["link_id"] = tmpl.link_id.astype(str)

    out_rows = []
    for wid, grp in tmpl.groupby("window_id"):
        w = widx[widx.window_id == wid].iloc[0]
        cond = w.condition
        t_horizon_end = grp.ts.max()
        # queue at T-5 (last history ts)
        h = hist[hist.window_id == wid]
        last = h[h.ts == h.ts.max()].drop_duplicates("link_id")
        last["q_now"] = (last.speed_kmh <= last.link_id.map(v_cut)).astype(int)
        q_now = dict(zip(last.link_id, last.q_now))
        if "ongoing" in cond:
            grp["queue_pred"] = grp.link_id.map(q_now).fillna(0).astype(int)
        else:
            # onset: predict the active cluster at T+30 (last step)
            t_origin = pd.Timestamp(w.forecast_origin)
            cluster = set(get_onset_cluster(panel, t_origin))
            grp["queue_pred"] = ((grp.link_id.isin(cluster)) & (grp.ts == t_horizon_end)).astype(int)
        out_rows.append(grp[["window_id", "timestamp", "link_id", "queue_pred"]])
    return pd.concat(out_rows, ignore_index=True)


if __name__ == "__main__":
    # self-eval on train windows
    import sys
    sys.path.insert(0, ".")
    # evaluate on train windows (derive truth from unmasked speed)
    rows = []
    for panel in SCORED_PANELS:
        widx = pd.read_csv(f"{ROOT}/task2/{panel}/train/window_index.csv")
        hist = pd.read_parquet(f"{ROOT}/task2/{panel}/train/window_history.parquet")
        tmpl = pd.read_csv(f"{ROOT}/task2/{panel}/train/sample_submission_queue.csv")
        links = pd.read_csv(f"{ROOT}/corridors/{panel}/network/links.csv", dtype={"link_id": str})
        v_cut = dict(zip(links.link_id, links.free_speed_kmh * 0.6))
        hist["ts"] = pd.to_datetime(hist.timestamp, utc=True)
        tmpl["ts"] = pd.to_datetime(tmpl.timestamp, utc=True)
        hist["link_id"] = hist.link_id.astype(str)
        tmpl["link_id"] = tmpl.link_id.astype(str)
        truth_files = {os.path.basename(p).replace("synthetic_mainline_", "").replace(".parquet", ""): p
                       for p in glob.glob(f"{ROOT}/corridors/{panel}/train/mainline_states/year_month=*/*.parquet")}
        for wid, grp in tmpl.groupby("window_id"):
            w = widx[widx.window_id == wid].iloc[0]
            cond = w.condition
            t_horizon_end = grp.ts.max()
            h = hist[hist.window_id == wid]
            last = h[h.ts == h.ts.max()].drop_duplicates("link_id")
            last["q_now"] = (last.speed_kmh <= last.link_id.map(v_cut)).astype(int)
            q_now = dict(zip(last.link_id, last.q_now))
            if "ongoing" in cond:
                pred = grp.link_id.map(q_now).fillna(0).astype(int)
            else:
                cluster = set(get_onset_cluster(panel, pd.Timestamp(w.forecast_origin)))
                pred = ((grp.link_id.isin(cluster)) & (grp.ts == t_horizon_end)).astype(int)
            # truth
            tf = truth_files.get(w.date.replace("-", "_"))
            day = pd.read_parquet(tf, columns=["timestamp", "link_id", "speed_kmh"])
            day["link_id"] = day.link_id.astype(str); day["ts"] = pd.to_datetime(day.timestamp, utc=True)
            fstart = pd.Timestamp(w.forecast_start); fend = pd.Timestamp(w.forecast_end)
            tr = day[(day.ts >= fstart) & (day.ts <= fend)].copy()
            tr["q"] = (tr.speed_kmh <= tr.link_id.map(v_cut)).astype(int)
            # align pred to truth
            m = grp.merge(tr[["ts", "link_id", "q"]], on=["ts", "link_id"], how="left")
            a = pred.to_numpy(); b = m.q.fillna(0).astype(int).to_numpy()
            inter = int((a & b).sum()); union = int((a | b).sum())
            iou = 1.0 if union == 0 else inter / union
            rows.append({"panel": panel, "condition": cond, "iou": iou, "n_true": int(b.sum())})
    res = pd.DataFrame(rows)
    print("=== 瓶颈簇 + persistence 队列（train 全 8 面板）===")
    print(f"overall: {res.iou.mean():.4f}  (persistence 基线 0.2812)")
    print(f"onset:   {res[res.condition=='queue_onset'].iou.mean():.4f}")
    print(f"ongoing: {res[res.condition=='queue_ongoing'].iou.mean():.4f}")
    print("\n按面板:")
    print(res.groupby("panel").iou.mean().round(4).to_string())
