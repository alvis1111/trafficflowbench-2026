"""Task 2 exploration: derive queue truth from train, score persistence, look at dynamics."""
import glob
import os
import numpy as np
import pandas as pd

PANEL = "D12_I5_N"
ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"

links = pd.read_csv(f"{ROOT}/corridors/{PANEL}/network/links.csv", dtype={"link_id": str})
v_cut = dict(zip(links.link_id, links.free_speed_kmh * 0.60))

widx = pd.read_csv(f"{ROOT}/task2/{PANEL}/train/window_index.csv")
hist = pd.read_parquet(f"{ROOT}/task2/{PANEL}/train/window_history.parquet")

truth_files = {}
for p in glob.glob(f"{ROOT}/corridors/{PANEL}/train/mainline_states/year_month=*/*.parquet"):
    date = os.path.basename(p).replace("synthetic_mainline_", "").replace(".parquet", "")
    truth_files[date] = p  # e.g. "2030_06_01"

def queue_truth(date, fstart, fend):
    tf = truth_files.get(date)
    day = pd.read_parquet(tf, columns=["timestamp", "link_id", "speed_kmh"])
    day["link_id"] = day.link_id.astype(str)
    day["ts"] = pd.to_datetime(day.timestamp, utc=True)
    h = day[(day.ts >= fstart) & (day.ts <= fend)].copy()
    h["queue_true"] = (h.speed_kmh <= h.link_id.map(v_cut)).astype(int)
    return h[["link_id", "ts", "queue_true"]]

rows = []
for _, w in widx.iterrows():
    fstart = pd.Timestamp(w.forecast_start); fend = pd.Timestamp(w.forecast_end)
    origin = pd.Timestamp(w.forecast_origin)
    truth = queue_truth(w.date.replace("-", "_"), fstart, fend)
    # persistence: queue status at the last history timestamp (T-5), repeated
    h = hist[hist.window_id == w.window_id].copy()
    h["ts"] = pd.to_datetime(h.timestamp, utc=True)
    org = h[h.ts == h.ts.max()][["link_id", "speed_kmh"]].copy()
    org["queue_now"] = (org.speed_kmh <= org.link_id.map(v_cut)).astype(int)
    pred = truth[["link_id", "ts"]].merge(org[["link_id", "queue_now"]], on="link_id", how="left")
    pred["queue_now"] = pred.queue_now.fillna(0).astype(int)
    a = pred.queue_now.to_numpy(); b = truth.queue_true.to_numpy()
    inter = int((a & b).sum()); union = int((a | b).sum())
    iou = 1.0 if union == 0 else inter / union
    rows.append({"window": w.window_id, "condition": w.condition,
                 "origin": str(origin), "iou": iou, "n_true": int(b.sum())})

res = pd.DataFrame(rows)
print("=== persistence baseline on TRAIN (%s) ===" % PANEL)
print(res.to_string(index=False))
print(f"\noverall IoU = {res.iou.mean():.4f}")
print(f"onset   IoU = {res[res.condition=='queue_onset'].iou.mean():.4f}")
print(f"ongoing IoU = {res[res.condition=='queue_ongoing'].iou.mean():.4f}")
