# TrafficFlowBench Release 1.0

Data package for the 2026 IEEE Big Data Cup. Ten directional freeway corridors
at five-minute resolution, with four scored tasks.

This file describes what is in the package. The tasks, the scoring rule and the
submission format are on the competition pages. Code, baselines and the four
evaluators are at https://github.com/jacky850/trafficflowbench-public

## What you upload

```
sample_submission.csv    fill in the values, upload this
submission_key.csv       what each submission_id refers to, download only
```

`sample_submission.csv` has one row for every value the leaderboard scores,
across both the validation and the private month:

```
submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow
1,state,0,0,0,0
```

`task` says which column that row is asking for: `state` wants `speed_kmh` and
`flow_vph`, `queue` wants `queue_pred` as 0 or 1, `odme` wants `path_flow`.
Leave the rest at 0. Never leave a cell empty; a submission with a blank value
is rejected before scoring.

There is no `physics` task in the submission. Task 3 is scored on your `state`
rows.

`submission_key.csv` maps every `submission_id` to the cell it names: task,
split, panel, timestamp, station, link, mask regime, window, path and zones.
Join on `submission_id` to work out what to predict. Do not upload it.

## Corridors

Five corridors, two directions each:

```
D7_I10_E    D7_I10_W
D7_I210_E   D7_I210_W
D7_I405_N   D7_I405_S
D12_I5_N    D12_I5_S
D12_I405_N  D12_I405_S
```

`D12_I405_N` and `D12_I405_S` have no Task 2 windows. They are scored on the
other three tasks.

## Layout

```
config/
  synthetic_release_v1.json     calendar and release contract
  corridors.json                the ten directional panels
  competition_contract.json     task definitions and score weights

corridors/<PANEL>/
  network/
    links.csv                   length, lanes, free speed, capacity
    fd_parameters.csv           per-link fundamental-diagram parameters
    lwr_mainline_topology.csv   which link feeds which
    ramp_attachment_map.csv     where ramps attach
    path_set.csv                paths, with origin and destination zones
    path_link_incidence.csv     which links each path uses
    base_od.csv                 the weak OD prior
  train/
    mainline_states/            observations, nothing removed
    mainline_states_masked/     the same, with the Task 1 targets blanked
    ramp_states/                on and off-ramp counts
  validation/  private/
    mainline_states_masked/
    ramp_states/
```

A ramp record carries the count the detector measured, its quality fields and
the ids. A ramp loop counts vehicles crossing the ramp; it does not measure the
freeway beside it, so there is no speed column.

```

task1/<PANEL>/<split>/  sample_submission_state.csv
task2/<PANEL>/<split>/  window_index.csv, window_history.parquet,
                        sample_submission_queue.csv
task4/<PANEL>/<split>/  synthetic_link_counts.csv, synthetic_weak_prior.csv,
                        sample_submission_path_flow.csv
```

The per-corridor templates under `task1/`, `task2/` and `task4/` are the
convenient shape for local work, and they are what the baselines in the GitHub
repository read. They also cover `train`, which the upload template does not.

## Splits

| Split | Dates | Days | Masked inputs | Unmasked observations |
| --- | --- | --- | --- | --- |
| `train` | 2030-06-01 to 2031-02-28 | 273 | yes | yes |
| `validation` | 2031-03 | 31 | yes | no |
| `private` | 2031-04 | 30 | yes | no |

The calendar is synthetic and matches no real period.

`train` is the only split that ships the unmasked layer, so it is the only one
you can score yourself on. Join `mainline_states_masked` to `mainline_states` on
`(timestamp, station_id, link_id)` to recover the answer for a blanked cell.

`validation` feeds the public leaderboard and `private` decides the final
ranking. They are different months, generated independently, with different
incidents. The two leaderboards will not agree.

## The Task 1 mask

`mainline_states_masked` is partitioned by `mask_regime`. R1 removes 20 percent
of eligible cells, R2 removes 30 percent, R3 removes 50 percent. Each calendar
day is published under exactly one regime.

At a blanked cell every measured channel is removed: `speed_kmh`, `flow_vph`,
`occupancy`, and the density derived from occupancy. A detector produces all of
them from the same measurement, so an outage takes all of them. The quality
fields stay.

Cells that fall inside a released Task 2 window history are not Task 1 targets,
because that history publishes raw observations. Deriving the mask yourself from
the published hash therefore gives a slightly larger set than the release
scores. Use the template.

## The gap after each Task 2 window

Around every scored Task 2 window the mainline observations stop: the
thirty-minute horizon itself, and the sixty minutes after it. This is
deliberate and it is not a detector outage. The horizon is what Task 2 asks you
to predict, so it cannot also ship as an observation, and without the buffer it
could be read back by interpolating across the gap from the far side.

Nothing in that span is scored on Task 2 except the horizon, and `window_index.csv`
already tells you where every window sits, so the gap gives nothing away. Task 1
targets are unaffected: the template remains the full and only list of the cells
Task 1 asks for, inside these spans as anywhere else.

## The quality rule

Every mainline and ramp cell carries `pct_observed`, the share of the
five-minute interval a detector actually reported.

```
is_imputed        = pct_observed < 100
is_score_eligible = pct_observed >= 75 and the required values are present
```

Only eligible cells are scored, in every task. About 76 percent qualify.
Ineligible cells are still released. An unavailable ramp observation is missing
evidence, not a measured flow of zero; treating it as zero will lower your
Task 3 score.

## Where the data comes from

The released records are synthetic. A calibrated traffic-flow model generated
them. Its parameters were measured from real freeway detector data on these same
corridors: free-flow speed, per-lane capacity, jam density, demand by weekday and
time of day, the size of the measurement noise, and how often detectors are
available.

No real detector record appears in the release. Nothing was shifted, resampled
or copied, and the released calendar matches no real period. The data obeys real
traffic physics and carries realistic measurement error, and there is no external
source to join it against.

## License

Subject to the Competition Rules. Network topology is derived in part from
OpenStreetMap: https://www.openstreetmap.org/copyright
