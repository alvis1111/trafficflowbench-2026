"""Apply physics corrections (FD projection + ramp-aware conservation) to the state."""
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
    return topo.sort_values("order_index", ascending=False).mainline_link_id.tolist()


def load_ramp_net(panel):
    """Net ramp flow (r_on - r_off) per (link_id, timestamp) for validation + private."""
    amap = pd.read_csv(f"{ROOT}/corridors/{panel}/network/ramp_attachment_map.csv", dtype=str)
    amap = amap.rename(columns={"nearest_mainline_link_id": "link_id"})
    frames = []
    for split in ["validation", "private"]:
        import glob
        for f in glob.glob(f"{ROOT}/corridors/{panel}/{split}/ramp_states/**/*.parquet", recursive=True):
            r = pd.read_parquet(f, columns=["timestamp", "ramp_link_id", "ramp_type", "flow_vph", "is_score_eligible"])
            r = r[r.is_score_eligible.astype(bool)]
            r["ramp_link_id"] = r.ramp_link_id.astype(str)
            r["ts"] = pd.to_datetime(r.timestamp, utc=True)
            r = r.merge(amap[["ramp_link_id", "link_id"]], on="ramp_link_id", how="inner")
            r["net"] = np.where(r.ramp_type == "OR", r.flow_vph, -r.flow_vph)  # on-ramp +, off-ramp -
            frames.append(r[["ts", "link_id", "net"]])
    if not frames:
        return {}
    rn = pd.concat(frames, ignore_index=True)
    rn = rn.groupby(["ts", "link_id"]).net.sum().reset_index()
    return {(str(r.link_id), r.ts): r.net for r in rn.itertuples()}


def soft_fd_projection(speed, flow, fd):
    vf = fd.free_speed_kmh.to_numpy(); cap = fd.capacity_vph.to_numpy()
    kcrit = fd.critical_density.to_numpy(); kjam = fd.k_jam.to_numpy(); w = fd.wave.to_numpy()
    valid = np.isfinite(speed) & np.isfinite(flow) & (speed > 1.0)
    k = np.where(valid, flow / np.maximum(speed, 1.0), 0.0)
    q_fd = np.where(k <= kcrit, vf * k, w * np.maximum(kjam - k, 0.0))
    reasonable = valid & (q_fd > 0) & (q_fd < cap * 1.3)
    flow_out = flow.copy()
    flow_out[reasonable] = (1.0 - FD_ALPHA) * flow[reasonable] + FD_ALPHA * q_fd[reasonable]
    return speed, flow_out


def conservation(link_ids, flow, order, ramp_net, ts):
    omap = {lid: i for i, lid in enumerate(order)}
    n = len(flow)
    pos = np.array([omap.get(str(link_ids[i]), -1) for i in range(n)])
    flow_out = flow.copy()
    idx = np.argsort(pos)
    for _ in range(CONS_PASSES):
        for j in range(n - 1):
            if pos[idx[j + 1]] != pos[idx[j]] + 1:
                continue
            q_up = flow_out[idx[j]]; q_dn = flow_out[idx[j + 1]]
            if not (np.isfinite(q_up) and np.isfinite(q_dn)):
                continue
            q_ramp = ramp_net.get((str(link_ids[idx[j]]), ts), 0.0)
            residual = q_dn - (q_up + q_ramp)
            corr = CONS_ALPHA * residual * 0.5
            flow_out[idx[j]] = q_up + corr
            flow_out[idx[j + 1]] = q_dn - corr
    return flow_out


def main():
    state = pd.read_csv("lgbm_state_all.csv", dtype={"link_id": str, "station_id": str, "panel": str})
    state["timestamp"] = pd.to_datetime(state.timestamp, utc=True)
    chunks = []
    for panel in PANELS:
        fd = load_fd(panel)
        order = load_order(panel)
        ramp_net = load_ramp_net(panel)
        sp = state[state.panel == panel].copy()
        fd_aligned = fd.reindex(sp.link_id)
        sp["speed_kmh"], sp["flow_vph"] = soft_fd_projection(sp.speed_kmh.to_numpy(), sp.flow_vph.to_numpy(), fd_aligned)
        for ts, grp in sp.groupby("timestamp"):
            sp.loc[grp.index, "flow_vph"] = conservation(grp.link_id.to_numpy(), grp.flow_vph.to_numpy(), order, ramp_net, ts)
        chunks.append(sp)
        print(f"  {panel} done", flush=True)
    out = pd.concat(chunks, ignore_index=True)
    out.to_csv("lgbm_state_phys_ramp.csv", index=False)
    print(f"wrote {len(out):,} rows to lgbm_state_phys_ramp.csv")


if __name__ == "__main__":
    main()
