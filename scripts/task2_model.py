"""Task 2 model: historical queue probability + persistence, tested on train windows."""
import glob
import os
import numpy as np
import pandas as pd

PANEL = "D12_I5_N"
ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"
N_SLOT = 288

links = pd.read_csv(f"{ROOT}/corridors/{PANEL}/network/links.csv", dtype={"link_id": str})
v_cut = dict(zip(links.link_id, links.free_speed_kmh * 0.60))
link_ids = sorted(links.link_id.unique())
link_idx = {l: i for i, l in enumerate(link_ids)}
n_link = len(link_ids)

# --- historical queue probability per (link, weekday, tod) ---
q_sum = np.zeros((n_link, 7, N_SLOT), dtype=np.float64)
q_cnt = np.zeros((n_link, 7, N_SLOT), dtype=np.float64)
train_files = sorted(glob.glob(f"{ROOT}/corridors/{PANEL}/train/mainline_states/year_month=*/*.parquet"))
for f in train_files:
    d = pd.read_parquet(f, columns=["timestamp", "link_id", "speed_kmh"])
    d["link_id"] = d.link_id.astype(str)
    d["ts"] = pd.to_datetime(d.timestamp, utc=True)
    d["dow"] = d.ts.dt.weekday
    d["tod"] = d.ts.dt.hour * 12 + d.ts.dt.minute // 5
    d["q"] = (d.speed_kmh <= d.link_id.map(v_cut)).astype(float)
    li = d.link_id.map(link_idx).to_numpy()
    np.add.at(q_sum, (li, d.dow.to_numpy(), d.tod.to_numpy()), d.q.to_numpy())
    np.add.at(q_cnt, (li, d.dow.to_numpy(), d.tod.to_numpy()), 1.0)
hist_p = np.divide(q_sum, np.maximum(q_cnt, 1))

# --- evaluate on train windows ---
widx = pd.read_csv(f"{ROOT}/task2/{PANEL}/train/window_index.csv")
hist = pd.read_parquet(f"{ROOT}/task2/{PANEL}/train/window_history.parquet")
truth_files = {}
for p in train_files:
    truth_files[os.path.basename(p).replace("synthetic_mainline_", "").replace(".parquet", "")] = p

def predict(win, th_onset, th_ongoing):
    fstart = pd.Timestamp(win.forecast_start); fend = pd.Timestamp(win.forecast_end)
    tf = truth_files[win.date.replace("-", "_")]
    day = pd.read_parquet(tf, columns=["timestamp", "link_id", "speed_kmh"])
    day["link_id"] = day.link_id.astype(str); day["ts"] = pd.to_datetime(day.timestamp, utc=True)
    hor = day[(day.ts >= fstart) & (day.ts <= fend)].copy()
    hor["dow"] = hor.ts.dt.weekday; hor["tod"] = hor.ts.dt.hour * 12 + hor.ts.dt.minute // 5
    hor["queue_true"] = (hor.speed_kmh <= hor.link_id.map(v_cut)).astype(int)
    # historical prob for horizon
    li = hor.link_id.map(link_idx).to_numpy()
    p = hist_p[li, hor.dow.to_numpy(), hor.tod.to_numpy()]
    hor["hist_p"] = p
    # persistence: queue at last history ts (T-5)
    h = hist[hist.window_id == win.window_id].copy()
    h["ts"] = pd.to_datetime(h.timestamp, utc=True)
    org = h[h.ts == h.ts.max()][["link_id", "speed_kmh"]].copy()
    org["q0"] = (org.speed_kmh <= org.link_id.map(v_cut)).astype(int)
    hor = hor.merge(org[["link_id", "q0"]], on="link_id", how="left")
    hor["q0"] = hor.q0.fillna(0).astype(int)
    # prediction
    th = th_onset if win.condition == "queue_onset" else th_ongoing
    if win.condition == "queue_onset":
        pred = (hor.hist_p > th).astype(int)
    else:
        pred = ((hor.q0 == 1) | (hor.hist_p > th)).astype(int)
    a = pred.to_numpy(); b = hor.queue_true.to_numpy()
    inter = int((a & b).sum()); union = int((a | b).sum())
    return (1.0 if union == 0 else inter / union), int(b.sum()), hor

print("=== grid search thresholds ===")
best = None
for to in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    for tg in [0.4, 0.5, 0.6, 0.7, 0.8]:
        ious, ons, ong = [], [], []
        for _, w in widx.iterrows():
            iou, n, _ = predict(w, to, tg)
            ious.append(iou)
            (ons if w.condition == "queue_onset" else ong).append(iou)
        ov = np.mean(ious)
        if best is None or ov > best[0]:
            best = (ov, to, tg, np.mean(ons), np.mean(ong))
print(f"best: overall={best[0]:.4f}  th_onset={best[1]}  th_ongoing={best[2]}  onset={best[3]:.4f}  ongoing={best[4]:.4f}")
print(f"(persistence baseline: overall=0.2812 onset=0.0 ongoing=0.5624)")
