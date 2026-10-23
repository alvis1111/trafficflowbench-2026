"""Demand calibration: denoise boundary conditions via weekday x tod profiles."""
import glob
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from ctm_engine import load_network, load_day, ctm_step

PANEL = "D12_I405_S"
ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"
PD = f"{ROOT}/corridors/{PANEL}"
SUB = 300

fd, on_r, off_r = load_network()
n = len(fd)
L = fd.length_km.to_numpy(); vf = fd.free_speed_kmh.to_numpy(); cap = fd.capacity_vph.to_numpy()
kjam = fd.k_jam.to_numpy(); w = fd.w.to_numpy()
order = fd.order_index.to_numpy()

# load all train days
all_q, all_ron, all_roff = [], [], []
dows, tods = None, None
for day_file in sorted(glob.glob(f"{PD}/train/mainline_states/year_month=*/*.parquet")):
    q, v, k, ron, roff = load_day(fd, on_r, off_r, day_file)
    ts = q.index
    dow = ts.weekday.to_numpy(); tod = (ts.hour * 12 + ts.minute // 5).to_numpy()
    all_q.append(q.to_numpy()); all_ron.append(ron.to_numpy()); all_roff.append(roff.to_numpy())
    dows = dow; tods = tod  # same calendar every day (synthetic)

Q = np.stack(all_q); RON = np.stack(all_ron); ROFF = np.stack(all_roff)
ndays, T, n = Q.shape
print(f"train days: {ndays}, timestamps: {T}, links: {n}")

# profile: mean origin demand + ramps by (weekday, tod)
origin_demand = np.zeros((7, 288))
ron_prof = np.zeros((7, 288, n)); roff_prof = np.zeros((7, 288, n))
cnt = np.zeros((7, 288))
for d in range(ndays):
    np.add.at(origin_demand, (dows, tods), Q[d, :, 0])  # upstream link flow
    np.add.at(ron_prof, (dows, tods), RON[d])
    np.add.at(roff_prof, (dows, tods), ROFF[d])
    np.add.at(cnt, (dows, tods), 1.0)
origin_demand = origin_demand / np.maximum(cnt, 1)
ron_prof = ron_prof / np.maximum(cnt[:, :, None], 1)
roff_prof = roff_prof / np.maximum(cnt[:, :, None], 1)

# validate: run CTM on a day using the PROFILE boundaries (not the noisy day)
# but keep the observed INITIAL state, then step forward and compare
errs = []
for d in [0, 5, 10, 20, 40]:
    k0 = Q[d, 0].copy()
    k0 = np.where(np.isfinite(k0), k0, 0.0)
    for t in range(T - 1):
        q_in = origin_demand[dows[t], tods[t]]
        r_on = ron_prof[dows[t], tods[t]]
        r_off = roff_prof[dows[t], tods[t]]
        for _ in range(SUB):
            k0 = ctm_step(k0, q_in, r_on, r_off, L, vf, cap, w, kjam)
        k_obs = Q[d, t + 1]
        m = np.isfinite(k_obs)
        errs.append(np.mean(np.abs(k0[m] - k_obs[m]) / np.maximum(k_obs[m], 1.0)))
errs = np.array(errs)
print(f"\n用历史均值 profile 做边界（初值用观测）：")
print(f"密度误差 中位数: {np.median(errs):.4f}  p90: {np.quantile(errs, 0.9):.4f}")
print(f"(对比：用当天带噪边界时 ~0.15，噪声水平 ~0.02)")
