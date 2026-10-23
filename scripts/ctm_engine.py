"""CTM engine with ramps, at a fine time step. Validates against one day.

The generator is a fine-grid LWR/CTM: cells = detector links, ramps as
sources/sinks, and the 5-min / 0.3-km release is the down-sampled output.
"""
import glob
import os
import numpy as np
import pandas as pd

PANEL = "D12_I405_S"
ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"
PD = f"{ROOT}/corridors/{PANEL}"
DT_H = 1.0 / 3600.0   # 1 second
SUB = 300             # 5 min / 1 s


def load_network(panel_dir=None):
    pd0 = panel_dir or PD
    fd = pd.read_csv(f"{pd0}/network/fd_parameters.csv", dtype={"link_id": str})
    # several stations can share one link; average per-link FD params
    fd = fd.groupby("link_id", as_index=False)[
        ["length_km", "lanes", "free_speed_kmh", "capacity_vph", "critical_density", "k_jam"]].mean()
    topo = pd.read_csv(f"{pd0}/network/lwr_mainline_topology.csv", dtype=str)
    topo["order_index"] = topo.order_index.astype(int)
    topo_sub = topo[["mainline_link_id", "order_index"]].drop_duplicates("mainline_link_id").rename(columns={"mainline_link_id": "link_id"})
    fd = fd.merge(topo_sub, on="link_id", how="inner")
    fd = fd.sort_values("order_index", ascending=False).reset_index(drop=True)  # upstream -> downstream
    fd["w"] = fd.capacity_vph / (fd.k_jam - fd.critical_density)  # backward wave speed (km/h)
    # ramp attachment -> mainline link
    amap = pd.read_csv(f"{pd0}/network/ramp_attachment_map.csv", dtype=str)
    amap = amap.rename(columns={"nearest_mainline_link_id": "link_id"})
    on_ramps = amap[amap.ramp_type == "OR"][["ramp_link_id", "link_id"]]
    off_ramps = amap[amap.ramp_type == "FR"][["ramp_link_id", "link_id"]]
    return fd, on_ramps, off_ramps


def load_day(fd, on_ramps, off_ramps, day_file, panel_dir=None):
    pd0 = panel_dir or PD
    # mainline observed
    d = pd.read_parquet(day_file, columns=["timestamp", "link_id", "speed_kmh", "flow_vph"])
    d["link_id"] = d.link_id.astype(str); d["ts"] = pd.to_datetime(d.timestamp, utc=True)
    q = d.pivot_table(index="ts", columns="link_id", values="flow_vph").reindex(columns=fd.link_id)
    v = d.pivot_table(index="ts", columns="link_id", values="speed_kmh").reindex(columns=fd.link_id)
    # fill detector-outage gaps (ineligible cells) by temporal interpolation
    q = q.interpolate(limit_direction="both").fillna(0.0)
    v = v.interpolate(limit_direction="both").fillna(50.0)
    k = q / v.clip(lower=1.0)
    # ramp flows -> mainline link sums (5-min)
    ramp_root = f"{pd0}/train/ramp_states"
    ramp_files = sorted(glob.glob(f"{ramp_root}/**/*.parquet", recursive=True))
    # only ramps for this day
    day_ramp = []
    for rf in ramp_files:
        r = pd.read_parquet(rf, columns=["timestamp", "ramp_link_id", "ramp_type", "flow_vph", "is_score_eligible"])
        r["ramp_link_id"] = r.ramp_link_id.astype(str)
        r = r[r.is_score_eligible.astype(bool)]
        day_ramp.append(r)
    r = pd.concat(day_ramp, ignore_index=True)
    r["ts"] = pd.to_datetime(r.timestamp, utc=True)
    r_on = r[r.ramp_type == "OR"].merge(on_ramps, on="ramp_link_id", how="inner")
    r_off = r[r.ramp_type == "FR"].merge(off_ramps, on="ramp_link_id", how="inner")
    ron = r_on.pivot_table(index="ts", columns="link_id", values="flow_vph", aggfunc="sum").reindex(columns=fd.link_id).fillna(0.0)
    roff = r_off.pivot_table(index="ts", columns="link_id", values="flow_vph", aggfunc="sum").reindex(columns=fd.link_id).fillna(0.0)
    # align to mainline timestamps
    idx = q.index
    ron = ron.reindex(idx).fillna(0.0)
    roff = roff.reindex(idx).fillna(0.0)
    return q, v, k, ron, roff


def demand(k, vf, cap):
    return np.minimum(vf * k, cap)


def supply(k, cap, w, kjam):
    return np.minimum(cap, w * np.maximum(kjam - k, 0.0))


def ctm_step(k, q_origin, r_on, r_off, L, vf, cap, w, kjam):
    k = np.asarray(k, dtype=float)
    r_on = np.asarray(r_on, dtype=float)
    r_off = np.asarray(r_off, dtype=float)
    L = np.asarray(L, dtype=float)
    vf = np.asarray(vf, dtype=float)
    cap = np.asarray(cap, dtype=float)
    w = np.asarray(w, dtype=float)
    kjam = np.asarray(kjam, dtype=float)
    n = len(k)
    q = np.zeros(n)
    for i in range(n - 1):
        q[i] = np.minimum(demand(k[i], vf[i], cap[i]), supply(k[i + 1], cap[i + 1], w[i + 1], kjam[i + 1]))
    q[n - 1] = demand(k[n - 1], vf[n - 1], cap[n - 1])
    k_new = np.zeros(n)
    for i in range(n):
        inflow = q_origin if i == 0 else q[i - 1]
        k_new[i] = k[i] + (DT_H / L[i]) * (inflow - q[i] + r_on[i] - r_off[i])
    return np.clip(k_new, 0.0, kjam)


def main():
    fd, on_ramps, off_ramps = load_network()
    n = len(fd)
    L = fd.length_km.to_numpy()
    vf = fd.free_speed_kmh.to_numpy()
    cap = fd.capacity_vph.to_numpy()
    kcrit = fd.critical_density.to_numpy()
    kjam = fd.k_jam.to_numpy()
    w = fd.w.to_numpy()
    print(f"links: {n}, min L = {L.min():.3f} km, Courant dt_max = {L.min()/vf.max()*3600:.1f}s (using 1s)")

    day_file = sorted(glob.glob(f"{PD}/train/mainline_states/year_month=*/*.parquet"))[0]
    q, v, k, ron, roff = load_day(fd, on_ramps, off_ramps, day_file)
    T = q.shape[0]
    print(f"day: {day_file.split('/')[-1]}, timestamps: {T}")

    # validate: step forward from observed k(t), compare to observed k(t+5min)
    errs = []
    for t in range(T - 1):
        k0 = k.iloc[t].to_numpy()
        q_in = q.iloc[t, 0]  # origin demand = upstream link's observed flow (vph)
        r_on = ron.iloc[t].to_numpy()
        r_off = roff.iloc[t].to_numpy()
        kt = k0.copy()
        for _ in range(SUB):
            kt = ctm_step(kt, q_in, r_on, r_off, L, vf, cap, w, kjam)
        k_next = k.iloc[t + 1].to_numpy()
        m = np.isfinite(k0) & np.isfinite(k_next)
        if m.sum() > 0:
            err = np.mean(np.abs(kt[m] - k_next[m]) / np.maximum(k_next[m], 1.0))
            errs.append(err)
    errs = np.array(errs)
    print(f"\nCTM 前向（细时间步 + 匝道）：")
    print(f"密度一步预测相对误差 中位数: {np.median(errs):.4f}")
    print(f"密度一步预测相对误差 p90: {np.quantile(errs, 0.9):.4f}")
    # baseline: observed step fluctuation
    step_fluct = np.nanmean(np.abs(k.diff().iloc[1:]) / k.iloc[1:].clip(lower=1.0))
    print(f"观测密度相邻步相对波动: {step_fluct:.4f}")


if __name__ == "__main__":
    main()
