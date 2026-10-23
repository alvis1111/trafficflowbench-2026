"""Queue forecast: CTM forward with trend+deviation demand forecast."""
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
SUB = 300

fd, on_r, off_r = load_network(PD)
n = len(fd)
L = fd.length_km.to_numpy(); vf = fd.free_speed_kmh.to_numpy(); cap = fd.capacity_vph.to_numpy()
kjam = fd.k_jam.to_numpy(); w = fd.w.to_numpy()
link_idx = {lid: i for i, lid in enumerate(fd.link_id)}
k_queue = w * kjam / (w + 0.6 * vf)  # density where speed = 0.6*vf

# --- build demand trend (weekday x tod) from train ---
days = sorted(glob.glob(f"{PD}/train/mainline_states/year_month=*/*.parquet"))
amap = pd.read_csv(f"{PD}/network/ramp_attachment_map.csv", dtype=str).rename(columns={"nearest_mainline_link_id": "link_id"})
ramp_files = sorted(glob.glob(f"{PD}/train/ramp_states/**/*.parquet", recursive=True))
rf = pd.concat([pd.read_parquet(f, columns=["timestamp", "ramp_link_id", "ramp_type", "flow_vph", "is_score_eligible"]) for f in ramp_files], ignore_index=True)
rf = rf[rf.is_score_eligible.astype(bool)]
rf["ts"] = pd.to_datetime(rf.timestamp, utc=True)
rf["dow"] = rf.ts.dt.weekday; rf["tod"] = rf.ts.dt.hour * 12 + rf.ts.dt.minute // 5
ron = rf[rf.ramp_type == "OR"].merge(on_r, on="ramp_link_id", how="inner")
roff = rf[rf.ramp_type == "FR"].merge(off_r, on="ramp_link_id", how="inner")

# origin demand trend: upstream link flow by (dow,tod)
origin_sum = np.zeros((7, 288)); origin_cnt = np.zeros((7, 288))
ron_sum = np.zeros((7, 288, n)); roff_sum = np.zeros((7, 288, n))
for day in days:
    d = pd.read_parquet(day, columns=["timestamp", "link_id", "flow_vph"])
    d["link_id"] = d.link_id.astype(str); d["ts"] = pd.to_datetime(d.timestamp, utc=True)
    d = d[d.link_id == fd.link_id.iloc[0]]  # upstream link
    d["dow"] = d.ts.dt.weekday; d["tod"] = d.ts.dt.hour * 12 + d.ts.dt.minute // 5
    np.add.at(origin_sum, (d.dow.to_numpy(), d.tod.to_numpy()), d.flow_vph.to_numpy())
    np.add.at(origin_cnt, (d.dow.to_numpy(), d.tod.to_numpy()), 1.0)
origin_trend = origin_sum / np.maximum(origin_cnt, 1)
# ramp trends
for (dow, tod), g in ron.groupby(["dow", "tod"]):
    for lid, v in g.groupby("link_id").flow_vph.sum().items():
        if lid in link_idx:
            ron_sum[dow, tod, link_idx[lid]] += v
for (dow, tod), g in roff.groupby(["dow", "tod"]):
    for lid, v in g.groupby("link_id").flow_vph.sum().items():
        if lid in link_idx:
            roff_sum[dow, tod, link_idx[lid]] += v
ron_cnt = np.zeros((7, 288, n)); roff_cnt = np.zeros((7, 288, n))
for (dow, tod), g in ron.groupby(["dow", "tod"]):
    for lid, c in g.groupby("link_id").size().items():
        if lid in link_idx: ron_cnt[dow, tod, link_idx[lid]] += c
for (dow, tod), g in roff.groupby(["dow", "tod"]):
    for lid, c in g.groupby("link_id").size().items():
        if lid in link_idx: roff_cnt[dow, tod, link_idx[lid]] += c
ron_trend = ron_sum / np.maximum(ron_cnt, 1)
roff_trend = roff_sum / np.maximum(roff_cnt, 1)

# --- windows ---
widx = pd.read_csv(f"{ROOT}/task2/{PANEL}/train/window_index.csv")
hist = pd.read_parquet(f"{ROOT}/task2/{PANEL}/train/window_history.parquet")

def queue_truth(date, fstart, fend):
    tf = None
    for p in days:
        if os.path.basename(p).replace("synthetic_mainline_", "").replace(".parquet", "") == date:
            tf = p; break
    day = pd.read_parquet(tf, columns=["timestamp", "link_id", "speed_kmh"])
    day["link_id"] = day.link_id.astype(str); day["ts"] = pd.to_datetime(day.timestamp, utc=True)
    h = day[(day.ts >= fstart) & (day.ts <= fend)].copy()
    h["q"] = (h.speed_kmh <= h.link_id.map(dict(zip(fd.link_id, 0.6 * vf)))).astype(int)
    return h

def forecast(win):
    h = hist[hist.window_id == win.window_id].copy()
    h["ts"] = pd.to_datetime(h.timestamp, utc=True)
    last = h[h.ts == h.ts.max()].drop_duplicates("link_id")
    spd = last.set_index("link_id").speed_kmh.reindex(fd.link_id).to_numpy()
    flw = last.set_index("link_id").flow_vph.reindex(fd.link_id).to_numpy()
    k0 = np.where(np.isfinite(flw) & np.isfinite(spd), flw / np.clip(spd, 1, None), np.where(np.isfinite(flw), flw / vf, 0.0))
    t0 = last.ts.iloc[0]
    dow0, tod0 = t0.weekday(), t0.hour * 12 + t0.minute // 5
    # deviation at T-5: observed - trend
    dev_origin = (flw[0] if np.isfinite(flw[0]) else origin_trend[dow0, tod0]) - origin_trend[dow0, tod0]
    dev_ron = ron[ron.ts == t0].groupby("link_id").flow_vph.sum().reindex(fd.link_id).fillna(0.0).to_numpy() - ron_trend[dow0, tod0]
    dev_roff = roff[roff.ts == t0].groupby("link_id").flow_vph.sum().reindex(fd.link_id).fillna(0.0).to_numpy() - roff_trend[dow0, tod0]
    k = k0.copy()
    preds = {}
    for step in range(6):
        tf = t0 + pd.Timedelta(minutes=5 * (step + 1))
        dowf, todf = tf.weekday(), tf.hour * 12 + tf.minute // 5
        q_origin = origin_trend[dowf, todf] + dev_origin
        r_on = ron_trend[dowf, todf] + dev_ron
        r_off = roff_trend[dowf, todf] + dev_roff
        for _ in range(SUB):
            k = ctm_step(k, q_origin, r_on, r_off, L, vf, cap, w, kjam)
        preds[tf] = (k > k_queue).astype(int)
    return k0, preds

rows = []
for _, wr in widx.iterrows():
    fstart = pd.Timestamp(wr.forecast_start); fend = pd.Timestamp(wr.forecast_end)
    truth = queue_truth(wr.date.replace("-", "_"), fstart, fend)
    _, preds = forecast(wr)
    a = []; b = []
    for _, tr in truth.iterrows():
        a.append(preds.get(tr.ts, np.zeros(n))[link_idx[tr.link_id]] if tr.link_id in link_idx else 0)
        b.append(tr.q)
    a = np.asarray(a, dtype=int); b = np.asarray(b, dtype=int)
    inter = int((a & b).sum()); union = int((a | b).sum())
    iou = 1.0 if union == 0 else inter / union
    rows.append({"condition": wr.condition, "iou": iou})

res = pd.DataFrame(rows)
print(f"=== 趋势+偏差需求预测 CTM 队列 ===")
print(f"overall: {res.iou.mean():.4f}  (persistence=0.2812)")
print(f"onset:   {res[res.condition=='queue_onset'].iou.mean():.4f}")
print(f"ongoing: {res[res.condition=='queue_ongoing'].iou.mean():.4f}  (persistence=0.5624)")
