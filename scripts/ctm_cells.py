"""Find the CTM cell structure by looking for piecewise-constant density along the corridor."""
import glob
import numpy as np
import pandas as pd

PANEL = "D12_I405_S"
ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"
PD = f"{ROOT}/corridors/{PANEL}"

fd = pd.read_csv(f"{PD}/network/fd_parameters.csv", dtype={"link_id": str})
links = pd.read_csv(f"{PD}/network/links.csv", dtype={"link_id": str})
topo = pd.read_csv(f"{PD}/network/lwr_mainline_topology.csv", dtype=str)

# spatial order: use milepost (from mainline_states) or order_index
# let's use order_index from topology: 0=downstream, higher=upstream. Traffic flows high->low.
topo["order_index"] = topo.order_index.astype(int)

# load one day, compute density k = q/v per link
day = sorted(glob.glob(f"{PD}/train/mainline_states/year_month=*/*.parquet"))[0]
d = pd.read_parquet(day, columns=["timestamp", "link_id", "milepost", "speed_kmh", "flow_vph"])
d["link_id"] = d.link_id.astype(str)
d["k"] = d.flow_vph / d.speed_kmh.clip(lower=1.0)
d = d.merge(topo[["mainline_link_id", "order_index", "length_km"]], left_on="link_id", right_on="mainline_link_id", how="inner")
d = d.sort_values(["order_index", "timestamp"])
d["order_index"] = d.order_index.astype(int)

# density profile at a few timestamps
print("=== 密度沿走廊分布（每个 link 一行，按 order_index 从下游到上游）===")
piv = d.pivot_table(index="order_index", columns="timestamp", values="k", aggfunc="mean")
# print a few time columns
ts_cols = sorted(piv.columns)[:6]
print("order_index |", " | ".join(str(c)[11:16] for c in ts_cols))
for oi in sorted(piv.index)[:20]:
    row = piv.loc[oi, ts_cols].round(1)
    print(f"{oi:5d}       |", " | ".join(f"{v:6.1f}" for v in row))

# correlation between adjacent links' density (to find cell boundaries)
print("\n=== 相邻 link 密度相关性（找 cell 边界）===")
# for each order_index, correlation with the next (upstream) link's density over time
kmat = piv.to_numpy()
orders = sorted(piv.index)
for i in range(len(orders) - 1):
    a = kmat[i]; b = kmat[i + 1]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() > 50:
        c = np.corrcoef(a[m], b[m])[0, 1]
        # flag low correlation as a potential cell boundary
        flag = " <-- boundary?" if c < 0.8 else ""
        print(f"  order {orders[i]:3d} -> {orders[i+1]:3d}: corr={c:.3f}{flag}")
