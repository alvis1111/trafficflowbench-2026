"""Apply physics corrections (soft FD projection + conservation) to the state submission."""
import numpy as np
import pandas as pd

ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S",
          "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]
FD_ALPHA = 0.15
CONS_ALPHA = 0.40
CONS_PASSES = 5


def load_fd(panel):
    fd = pd.read_csv(f"{ROOT}/corridors/{panel}/network/fd_parameters.csv", dtype={"link_id": str})
    fd = fd.groupby("link_id", as_index=False)[["free_speed_kmh", "capacity_vph", "critical_density", "k_jam"]].mean()
    fd["wave"] = fd.capacity_vph / (fd.k_jam - fd.critical_density)
    return fd.set_index("link_id")


def load_order(panel):
    topo = pd.read_csv(f"{ROOT}/corridors/{panel}/network/lwr_mainline_topology.csv", dtype=str)
    topo = topo[["mainline_link_id", "order_index"]].drop_duplicates("mainline_link_id")
    topo["order_index"] = topo.order_index.astype(int)
    # upstream -> downstream = desc order_index
    topo = topo.sort_values("order_index", ascending=False)
    return topo.mainline_link_id.tolist()


def soft_fd_projection(speed, flow, fd):
    vf = fd.free_speed_kmh.to_numpy(); cap = fd.capacity_vph.to_numpy()
    kcrit = fd.critical_density.to_numpy(); kjam = fd.k_jam.to_numpy(); w = fd.wave.to_numpy()
    valid = np.isfinite(speed) & np.isfinite(flow) & (speed > 1.0)
    k = np.where(valid, flow / np.maximum(speed, 1.0), 0.0)
    is_free = k <= kcrit
    q_fd = np.where(is_free, vf * k, w * np.maximum(kjam - k, 0.0))
    reasonable = valid & (q_fd > 0) & (q_fd < cap * 1.3)
    flow_out = flow.copy()
    flow_out[reasonable] = (1.0 - FD_ALPHA) * flow[reasonable] + FD_ALPHA * q_fd[reasonable]
    return speed, flow_out


def conservation_correction(link_ids, flow, order):
    # order: list of links upstream->downstream. order map: link -> position.
    omap = {lid: i for i, lid in enumerate(order)}
    n = len(flow)
    pos = np.array([omap.get(str(link_ids[i]), -1) for i in range(n)])
    flow_out = flow.copy()
    # group by timestamp handled by caller: here flow is one timestamp slice
    # sort by position
    idx = np.argsort(pos)
    for _ in range(CONS_PASSES):
        for j in range(n - 1):
            if pos[idx[j + 1]] != pos[idx[j]] + 1:
                continue
            q_up = flow_out[idx[j]]; q_dn = flow_out[idx[j + 1]]
            if not (np.isfinite(q_up) and np.isfinite(q_dn)):
                continue
            residual = q_dn - q_up
            corr = CONS_ALPHA * residual * 0.5
            flow_out[idx[j]] = q_up + corr
            flow_out[idx[j + 1]] = q_dn - corr
    return flow_out


def main():
    state = pd.read_csv("lgbm_state_all.csv", dtype={"link_id": str, "station_id": str, "panel": str})
    state["timestamp"] = pd.to_datetime(state.timestamp, utc=True)
    print(f"loaded {len(state):,} rows", flush=True)

    chunks = []
    for panel in PANELS:
        fd = load_fd(panel)
        order = load_order(panel)
        sp = state[state.panel == panel].copy()
        # FD projection (per cell, vectorized)
        fd_aligned = fd.reindex(sp.link_id)
        sp["speed_kmh"], sp["flow_vph"] = soft_fd_projection(
            sp.speed_kmh.to_numpy(), sp.flow_vph.to_numpy(), fd_aligned)
        # conservation (per timestamp)
        for ts, grp in sp.groupby("timestamp"):
            sp.loc[grp.index, "flow_vph"] = conservation_correction(
                grp.link_id.to_numpy(), grp.flow_vph.to_numpy(), order)
        chunks.append(sp)
        print(f"  {panel} done", flush=True)

    out = pd.concat(chunks, ignore_index=True)
    out.to_csv("lgbm_state_phys.csv", index=False)
    print(f"wrote {len(out):,} rows to lgbm_state_phys.csv")


if __name__ == "__main__":
    main()
