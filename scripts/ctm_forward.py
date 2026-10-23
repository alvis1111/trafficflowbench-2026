"""CTM Stage 1: minimal forward simulation to validate the model.

Given an initial density field and boundary demand, step the Cell Transmission
Model forward and compare the predicted flow/speed to the observations.
"""
import glob
import numpy as np
import pandas as pd

PANEL = "D12_I405_S"
ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"
PD = f"{ROOT}/corridors/{PANEL}"
DT = 5.0 / 60.0  # 5 min in hours

# --- network ---
fd = pd.read_csv(f"{PD}/network/fd_parameters.csv", dtype={"link_id": str})
fd = fd.groupby("link_id", as_index=False).agg(
    vf=("free_speed_kmh", "mean"), cap=("capacity_vph", "mean"),
    kcrit=("critical_density", "mean"), kjam=("k_jam", "mean"), lanes=("lanes", "mean"))
links = pd.read_csv(f"{PD}/network/links.csv", dtype={"link_id": str})
fd = fd.merge(links[["link_id", "length_km"]], on="link_id", how="left")
topo = pd.read_csv(f"{PD}/network/lwr_mainline_topology.csv", dtype=str)
order = topo[["mainline_link_id", "order_index"]].drop_duplicates()
order["order_index"] = order.order_index.astype(int)
fd = fd.merge(order.rename(columns={"mainline_link_id": "link_id"}), on="link_id", how="inner")
# order: 0 = downstream, higher = upstream. Sort upstream -> downstream (desc order).
fd = fd.sort_values("order_index", ascending=False).reset_index(drop=True)
fd["w"] = (fd.cap / fd.lanes) / ((fd.kjam / fd.lanes) - (fd.kcrit / fd.lanes))  # backward wave speed (km/h)

n = len(fd)
link_id = fd.link_id.to_numpy()
L = fd.length_km.to_numpy()
lanes = fd.lanes.to_numpy()
vf = fd.vf.to_numpy()
cap = fd.cap.to_numpy()          # total capacity (vph)
kcrit = fd.kcrit.to_numpy()      # total critical density (veh/km)
kjam = fd.kjam.to_numpy()        # total jam density
w = fd.w.to_numpy()

# --- one day ---
day_file = sorted(glob.glob(f"{PD}/train/mainline_states/year_month=*/*.parquet"))[0]
d = pd.read_parquet(day_file, columns=["timestamp", "link_id", "speed_kmh", "flow_vph"])
d["link_id"] = d.link_id.astype(str); d["ts"] = pd.to_datetime(d.timestamp, utc=True)
# pivot observed flow/speed by (ts, link), in the CTM order
q_obs = d.pivot_table(index="ts", columns="link_id", values="flow_vph").reindex(columns=link_id).to_numpy()
v_obs = d.pivot_table(index="ts", columns="link_id", values="speed_kmh").reindex(columns=link_id).to_numpy()
k_obs = q_obs / np.clip(v_obs, 1.0, None)  # observed density (total)

T = q_obs.shape[0]

def demand(k):
    return np.minimum(vf * k, cap)

def supply(k):
    return np.minimum(cap, w * np.maximum(kjam - k, 0.0))

def ctm_step(k, q_in_origin):
    """One 5-min CTM step. Traffic flows index 0 (upstream) -> n-1 (downstream)."""
    # q[i] = flow from link i to link i+1 (downstream neighbour)
    q = np.zeros(n)
    for i in range(n - 1):
        q[i] = np.minimum(demand(k)[i], supply(k)[i + 1])
    q[n - 1] = demand(k)[n - 1]  # most downstream link drains to destination
    k_new = np.zeros(n)
    for i in range(n):
        inflow = q_in_origin if i == 0 else q[i - 1]
        outflow = q[i]
        k_new[i] = k[i] + (DT / L[i]) * (inflow - outflow)
    return np.clip(k_new, 0.0, kjam)

# --- validate: step forward from observed k(t), compare to observed k(t+dt) ---
errs = []
for t in range(T - 1):
    k0 = k_obs[t].copy()
    # origin demand = observed flow at the most upstream link (index 0)
    q_in = q_obs[t, 0]
    k_pred = ctm_step(k0, q_in)
    k_next = k_obs[t + 1]
    m = np.isfinite(k0) & np.isfinite(k_next) & np.isfinite(k_pred)
    if m.sum() > 0:
        err = np.mean(np.abs(k_pred[m] - k_next[m]) / np.maximum(k_next[m], 1.0))
        errs.append(err)

errs = np.array(errs)
print(f"CTM 前向仿真（无匝道）：")
print(f"密度一步预测相对误差 中位数: {np.median(errs):.4f}")
print(f"密度一步预测相对误差 p90: {np.quantile(errs, 0.9):.4f}")
print(f"(对比：观测密度自身相邻步的波动水平)")
k_obs_finite = np.where(np.isfinite(k_obs), k_obs, np.nan)
step_diff = np.nanmean(np.abs(np.diff(k_obs_finite, axis=0)) / np.maximum(np.nanmean(k_obs_finite, axis=0), 1.0))
print(f"观测密度相邻步相对波动: {step_diff:.4f}")
