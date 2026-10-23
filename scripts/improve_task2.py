"""Task 2 improved: historical queue probability + persistence.

For each scored panel, learn the per-(link, weekday, time-of-day) queue
probability from train, then predict each horizon cell as:
  onset   -> hist_p > th_onset
  ongoing -> (queue at T-5) OR (hist_p > th_ongoing)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_SLOT = 288
# 8 scored panels (D12_I405 excluded from Task 2)
SCORED_PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W",
                 "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]


def v_cut_map(panel_dir: Path) -> dict:
    links = pd.read_csv(panel_dir / "network" / "links.csv", dtype={"link_id": str})
    return dict(zip(links.link_id, links.free_speed_kmh * 0.60))


def build_hist_p(panel_dir: Path, v_cut: dict):
    links = pd.read_csv(panel_dir / "network" / "links.csv", dtype={"link_id": str})
    link_ids = sorted(links.link_id.unique())
    link_idx = {l: i for i, l in enumerate(link_ids)}
    n = len(link_ids)
    qs = np.zeros((n, 7, N_SLOT)); qc = np.zeros((n, 7, N_SLOT))
    for f in sorted((panel_dir / "train" / "mainline_states").glob("**/*.parquet")):
        d = pd.read_parquet(f, columns=["timestamp", "link_id", "speed_kmh"])
        d["link_id"] = d.link_id.astype(str)
        d["ts"] = pd.to_datetime(d.timestamp, utc=True)
        dow = d.ts.dt.weekday.to_numpy()
        tod = (d.ts.dt.hour * 12 + d.ts.dt.minute // 5).to_numpy()
        q = (d.speed_kmh <= d.link_id.map(v_cut)).astype(float).to_numpy()
        li = d.link_id.map(link_idx).to_numpy()
        np.add.at(qs, (li, dow, tod), q)
        np.add.at(qc, (li, dow, tod), 1.0)
    return link_idx, np.divide(qs, np.maximum(qc, 1))


def task2_parts(root: Path, split: str, name: str) -> list:
    return sorted(p / split / name for p in (root / "task2").glob("*")
                  if (p / split / name).exists())


def build_split(panel, release, split, th_onset, th_ongoing, output, write_header):
    panel_dir = release / "corridors" / panel
    v_cut = v_cut_map(panel_dir)
    link_idx, hist_p = build_hist_p(panel_dir, v_cut)

    windows = pd.concat([pd.read_csv(p) for p in task2_parts(release, split, "window_index.csv")])
    windows = windows[windows.panel == panel]
    if windows.empty:
        return 0
    template = pd.concat([pd.read_csv(p, usecols=["window_id", "timestamp", "link_id"])
                          for p in task2_parts(release, split, "sample_submission_queue.csv")])
    template = template[template.window_id.isin(set(windows.window_id))].copy()
    hist = pd.concat([pd.read_parquet(p, columns=["window_id", "timestamp", "link_id", "speed_kmh"])
                      for p in task2_parts(release, split, "window_history.parquet")])
    hist = hist[hist.window_id.isin(set(windows.window_id))].copy()

    cond = dict(zip(windows.window_id.astype(str), windows.condition))
    # persistence: queue at last history timestamp per window
    hist["ts"] = pd.to_datetime(hist.timestamp, utc=True)
    hist["link_id"] = hist.link_id.astype(str)
    last = hist.loc[hist.groupby("window_id").ts.idxmax(), ["window_id", "link_id", "speed_kmh"]].copy()
    last["q0"] = (last.speed_kmh <= last.link_id.map(v_cut)).astype(int)
    last = last[["window_id", "link_id", "q0"]]

    template["ts"] = pd.to_datetime(template.timestamp, utc=True)
    template["link_id"] = template.link_id.astype(str)
    template["dow"] = template.ts.dt.weekday.to_numpy()
    template["tod"] = (template.ts.dt.hour * 12 + template.ts.dt.minute // 5).to_numpy()
    li = template.link_id.map(link_idx).to_numpy()
    template["hist_p"] = hist_p[li, template.dow.to_numpy(), template.tod.to_numpy()]
    template = template.merge(last, on=["window_id", "link_id"], how="left")
    template["q0"] = template.q0.fillna(0).astype(int)

    preds = []
    for wid, g in template.groupby("window_id"):
        c = cond.get(str(wid))
        if c is None:
            continue
        th = th_onset if c == "queue_onset" else th_ongoing
        p = ((g.hist_p > th).to_numpy() if c == "queue_onset"
             else ((g.q0 == 1) | (g.hist_p > th)).to_numpy())
        preds.append(pd.DataFrame({"window_id": wid, "timestamp": g.timestamp,
                                   "link_id": g.link_id, "queue_pred": p.astype(int)}))
    out = pd.concat(preds, ignore_index=True)
    out.to_csv(output, mode="w" if write_header else "a", header=write_header, index=False)
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", type=Path, default=Path("C:/Users/GEM07/OneDrive/Desktop/kaggle_public"))
    ap.add_argument("--split", choices=["validation", "private"], default="validation")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--th-onset", type=float, default=0.6)
    ap.add_argument("--th-ongoing", type=float, default=0.4)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    total, first = 0, True
    for panel in SCORED_PANELS:
        print(f"[task2] {panel} ({args.split})", flush=True)
        n = build_split(panel, args.release_root.resolve(), args.split,
                        args.th_onset, args.th_ongoing, args.output.resolve(), first)
        total += n
        first = False
        print(f"  wrote {n:,} rows", flush=True)
    print(f"Wrote {total:,} rows to {args.output}")


if __name__ == "__main__":
    main()
