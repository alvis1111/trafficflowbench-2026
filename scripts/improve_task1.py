"""Task 1 improved: historical prior + same-day temporal interpolation blend.

The baseline predicts each masked cell with the link's own weekday x time-of-day
mean.  That misses the actual traffic state of *this* day (incidents, demand
surges, queue formation).  This builder adds a same-day temporal estimate: for
each link it linearly interpolates the day's observed (unmasked) speed and flow
across time, then blends the interpolation with the historical prior.

    pred = alpha * interp + (1 - alpha) * prior
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the external repo's task1 helpers importable.
sys.path.insert(0, str(Path(__file__).resolve().parent / "repo" / "src"))
from task1.baseline_task1_historical_mean import (
    build_profile,
    check_release_root,
    masked_files,
    slot_values,
)

OUTPUT_COLUMNS = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]


def _interp(group: pd.Series) -> pd.Series:
    """Linear temporal interpolation, then fill edges with nearest observation."""
    return group.interpolate(method="linear", limit_direction="both")


def build_panel(panel: str, release: Path, split: str, output: Path,
                alpha: float, write_header: bool) -> int:
    check_release_root(release, panel)
    panel_dir = release / "corridors" / panel
    speed_profile, flow_profile, profile_counts = build_profile(panel, panel_dir)
    link_index = speed_profile["link_index"]
    n_slots = 7 * 288

    template = pd.read_csv(
        release / "task1" / panel / split / "sample_submission_state.csv",
        usecols=["timestamp", "station_id", "link_id"], dtype=str,
    )
    template_keys = set(zip(template.timestamp, template.station_id, template.link_id))

    rows_written = 0
    for path in masked_files(panel_dir, split):
        frame = pd.read_parquet(
            path,
            columns=["date", "timestamp", "station_id", "link_id", "speed_kmh",
                     "flow_vph", "is_score_eligible", "mask_regime"],
        )
        frame["station_id"] = frame.station_id.astype(str)
        frame["link_id"] = frame.link_id.astype(str)
        frame = frame.sort_values(["link_id", "timestamp"]).reset_index(drop=True)

        # Historical prior, same slot indexing as the baseline.
        weekday, tod = slot_values(frame)
        slot = weekday * 288 + tod
        li = frame.link_id.map(link_index).fillna(-1).to_numpy(dtype=np.int64)
        known = li >= 0
        safe_li = np.where(known, li, 0)
        base_speed = speed_profile["mean"][safe_li, slot]
        base_flow = flow_profile["mean"][safe_li, slot]
        base_speed = np.where(
            profile_counts["speed_count"][safe_li, slot] == 0,
            speed_profile["fallback"][safe_li], base_speed)
        base_flow = np.where(
            profile_counts["flow_count"][safe_li, slot] == 0,
            flow_profile["fallback"][safe_li], base_flow)

        # Speed: raw same-day temporal interpolation (lightly blended with prior).
        interp_speed = frame.groupby("link_id")["speed_kmh"].transform(_interp).to_numpy(dtype=float)
        # Flow: residual interpolation (prior + interpolated day-specific residual).
        #   Flow has a strong diurnal shape, so interpolating the residual (which is
        #   flatter) beats interpolating raw flow across a mask gap.
        frame["resid_flow"] = frame["flow_vph"].to_numpy(dtype=float) - base_flow
        interp_resid_flow = frame.groupby("link_id")["resid_flow"].transform(_interp).to_numpy(dtype=float)

        pred_speed = np.where(np.isfinite(interp_speed),
                              alpha * interp_speed + (1 - alpha) * base_speed, base_speed)
        pred_flow = np.where(np.isfinite(interp_resid_flow),
                             base_flow + interp_resid_flow, base_flow)
        pred_speed = np.maximum(pred_speed, 0.0)
        pred_flow = np.maximum(pred_flow, 0.0)

        blanked = np.fromiter(
            (k in template_keys for k in zip(frame.timestamp.astype(str),
                                             frame.station_id, frame.link_id)),
            dtype=bool, count=len(frame))

        for regime in pd.unique(frame.mask_regime.astype(str)):
            target = blanked & (frame.mask_regime.astype(str) == regime).to_numpy()
            mask = target & np.isfinite(pred_speed) & np.isfinite(pred_flow)
            if not mask.any():
                continue
            out = pd.DataFrame(
                {
                    "panel": panel,
                    "timestamp": frame.loc[mask, "timestamp"].astype(str).to_numpy(),
                    "station_id": frame.loc[mask, "station_id"].to_numpy(),
                    "link_id": frame.loc[mask, "link_id"].to_numpy(),
                    "mask_regime": regime,
                    "speed_kmh": pred_speed[mask],
                    "flow_vph": pred_flow[mask],
                },
                columns=OUTPUT_COLUMNS,
            )
            out.to_csv(output, mode="w" if write_header else "a", header=write_header, index=False)
            write_header = False
            rows_written += len(out)
    return rows_written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", type=Path,
                    default=Path("C:/Users/GEM07/OneDrive/Desktop/kaggle_public"))
    ap.add_argument("--split", choices=["train", "validation", "private"], default="train")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--panel", action="append")
    ap.add_argument("--alpha", type=float, default=0.85)
    args = ap.parse_args()

    manifest = json.loads((Path(__file__).resolve().parent / "repo" / "config" / "corridors.json").read_text())
    panels = [p["corridor_id"] for p in manifest["panels"]]
    if args.panel:
        panels = [p for p in panels if p in set(args.panel)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    total, first = 0, True
    for panel in panels:
        print(f"[improved] {panel}", flush=True)
        n = build_panel(panel, args.release_root.resolve(), args.split, args.output.resolve(),
                        args.alpha, first)
        total += n
        first = False
        print(f"  wrote {n:,} rows", flush=True)
    print(f"Wrote {total:,} rows to {args.output}")


if __name__ == "__main__":
    main()
