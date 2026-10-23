"""Build Task 2 queue submission (bottleneck clusters + persistence) for val+private."""
import pandas as pd

ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"
SCORED_PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W",
                 "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]

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


def build_split(split, out_path):
    outs = []
    for panel in SCORED_PANELS:
        widx = pd.read_csv(f"{ROOT}/task2/{panel}/{split}/window_index.csv")
        hist = pd.read_parquet(f"{ROOT}/task2/{panel}/{split}/window_history.parquet")
        tmpl = pd.read_csv(f"{ROOT}/task2/{panel}/{split}/sample_submission_queue.csv")
        links = pd.read_csv(f"{ROOT}/corridors/{panel}/network/links.csv", dtype={"link_id": str})
        v_cut = dict(zip(links.link_id, links.free_speed_kmh * 0.6))
        hist["ts"] = pd.to_datetime(hist.timestamp, utc=True)
        tmpl["ts"] = pd.to_datetime(tmpl.timestamp, utc=True)
        hist["link_id"] = hist.link_id.astype(str)
        tmpl["link_id"] = tmpl.link_id.astype(str)
        for wid, grp in tmpl.groupby("window_id"):
            w = widx[widx.window_id == wid].iloc[0]
            cond = w.condition
            t_horizon_end = grp.ts.max()
            h = hist[hist.window_id == wid]
            last = h[h.ts == h.ts.max()].drop_duplicates("link_id")
            last["q_now"] = (last.speed_kmh <= last.link_id.map(v_cut)).astype(int)
            q_now = dict(zip(last.link_id, last.q_now))
            if "ongoing" in cond:
                grp["queue_pred"] = grp.link_id.map(q_now).fillna(0).astype(int)
            else:
                cluster = set(get_onset_cluster(panel, pd.Timestamp(w.forecast_origin)))
                grp["queue_pred"] = ((grp.link_id.isin(cluster)) & (grp.ts == t_horizon_end)).astype(int)
            outs.append(grp[["window_id", "timestamp", "link_id", "queue_pred"]])
        print(f"  {panel} done", flush=True)
    out = pd.concat(outs, ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out):,} rows to {out_path}")


if __name__ == "__main__":
    import sys
    for split in ["validation", "private"]:
        build_split(split, f"queue_bn_{split}.csv")
