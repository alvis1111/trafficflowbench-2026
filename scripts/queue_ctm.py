"""Queue forecast with CTM: estimate state+demand from history, run forward, predict queue."""
import glob
import os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from ctm_engine import load_network, ctm_step

PANEL = "D12_I5_N"
ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"
PD = f"{ROOT}/corridors/{PANEL}"
SUB = 300  # 5 min in 1s steps

fd, on_r, off_r = load_network(PD)
n = len(fd)
L = fd.length_km.to_numpy(); vf = fd.free_speed_kmh.to_numpy(); cap = fd.capacity_vph.to_numpy()
kcrit = fd.critical_density.to_numpy(); kjam = fd.k_jam.to_numpy(); w = fd.w.to_numpy()
link_idx = {lid: i for i, lid in enumerate(fd.link_id)}
# queue threshold: speed <= 0.6*vf  ->  density k where FD-speed drops to 0.6*vf
# in congestion speed = w*(kjam-k)/k, set = 0.6*vf -> k = w*kjam/(w+0.6*vf)
k_queue = w * kjam / (w + 0.6 * vf)

widx = pd.read_csv(f"{ROOT}/task2/{PANEL}/train/window_index.csv")
hist = pd.read_parquet(f"{ROOT}/task2/{PANEL}/train/window_history.parquet")
# ramp flows for this panel (train), mapped to mainline
amap = pd.read_csv(f"{PD}/network/ramp_attachment_map.csv", dtype=str).rename(columns={"nearest_mainline_link_id": "link_id"})
ramp_files = sorted(glob.glob(f"{PD}/train/ramp_states/**/*.parquet", recursive=True))
rframes = []
for rf in ramp_files:
    r = pd.read_parquet(rf, columns=["timestamp", "ramp_link_id", "ramp_type", "flow_vph", "is_score_eligible"])
    r = r[r.is_score_eligible.astype(bool)]
    rframes.append(r)
ramps = pd.concat(rframes, ignore_index=True)
ramps["ts"] = pd.to_datetime(ramps.timestamp, utc=True)
ron = ramps[ramps.ramp_type == "OR"].merge(on_r, on="ramp_link_id", how="inner")
roff = ramps[ramps.ramp_type == "FR"].merge(off_r, on="ramp_link_id", how="inner")

def queue_truth(date, fstart, fend):
    # derive queue from unmasked train speed
    tf = None
    for p in glob.glob(f"{PD}/train/mainline_states/year_month=*/*.parquet"):
        if os.path.basename(p).replace("synthetic_mainline_", "").replace(".parquet", "") == date:
            tf = p; break
    day = pd.read_parquet(tf, columns=["timestamp", "link_id", "speed_kmh"])
    day["link_id"] = day.link_id.astype(str); day["ts"] = pd.to_datetime(day.timestamp, utc=True)
    h = day[(day.ts >= fstart) & (day.ts <= fend)].copy()
    h["q"] = (h.speed_kmh <= h.link_id.map(dict(zip(fd.link_id, 0.6 * vf)))).astype(int)
    return h

def ctm_forecast(win):
    # history: 60 min mainline state, estimate density at last ts (T-5)
    h = hist[hist.window_id == win.window_id].copy()
    h["ts"] = pd.to_datetime(h.timestamp, utc=True)
    h["link_id"] = h.link_id.astype(str)
    last = h[h.ts == h.ts.max()].drop_duplicates("link_id")
    # density from observed speed/flow at T-5 (fill missing via FD)
    spd = last.set_index("link_id").speed_kmh.reindex(fd.link_id).to_numpy()
    flw = last.set_index("link_id").flow_vph.reindex(fd.link_id).to_numpy()
    k0 = np.where(np.isfinite(flw) & np.isfinite(spd), flw / np.clip(spd, 1, None),
                  np.where(np.isfinite(flw), flw / vf, 0.0))
    # origin demand = upstream link flow at T-5 (persisted)
    q_origin = flw[0] if np.isfinite(flw[0]) else 0.0
    # ramp flows at T-5 (persisted)
    ron_t = ron[(ron.ts == last.ts.iloc[0])].drop_duplicates("link_id").set_index("link_id").flow_vph.reindex(fd.link_id).fillna(0.0).to_numpy()
    roff_t = roff[(roff.ts == last.ts.iloc[0])].drop_duplicates("link_id").set_index("link_id").flow_vph.reindex(fd.link_id).fillna(0.0).to_numpy()
    # run forward 6 cells (30 min), persisting demand
    k = k0.copy()
    preds = {}
    for step in range(6):
        for _ in range(SUB):
            k = ctm_step(k, q_origin, ron_t, roff_t, L, vf, cap, w, kjam)
        # queue at this future time = k > k_queue
        preds[last.ts.iloc[0] + pd.Timedelta(minutes=5 * (step + 1))] = (k > k_queue).astype(int)
    return k0, preds

# evaluate
rows = []
for _, wr in widx.iterrows():
    fstart = pd.Timestamp(wr.forecast_start); fend = pd.Timestamp(wr.forecast_end)
    truth = queue_truth(wr.date.replace("-", "_"), fstart, fend)
    _, preds = ctm_forecast(wr)
    # build prediction aligned to truth
    a = []; b = []
    for _, tr in truth.iterrows():
        key = tr.ts
        a.append(preds.get(key, np.zeros(n))[link_idx[tr.link_id]] if tr.link_id in link_idx else 0)
        b.append(tr.q)
    a = np.asarray(a, dtype=int); b = np.asarray(b, dtype=int)
    inter = int((a & b).sum()); union = int((a | b).sum())
    iou = 1.0 if union == 0 else inter / union
    rows.append({"window": wr.window_id, "condition": wr.condition, "iou": iou, "n_true": int(b.sum())})

res = pd.DataFrame(rows)
print("=== CTM 队列预测 vs persistence ===")
print(res.to_string(index=False))
print(f"\noverall: CTM={res.iou.mean():.4f}  (persistence=0.2812)")
print(f"onset:   CTM={res[res.condition=='queue_onset'].iou.mean():.4f}  (persistence=0.0)")
print(f"ongoing: CTM={res[res.condition=='queue_ongoing'].iou.mean():.4f}  (persistence=0.5624)")
