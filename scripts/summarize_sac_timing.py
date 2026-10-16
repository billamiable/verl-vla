#!/usr/bin/env python3
"""Summarize per-step timing from a SAC/DSRL training console log.

Each logged step is one console line of the form ``step:N - key:value - ...``.
The script separates the steps that run a rollout (the first step of every
rollout window) from the plain update-only steps, because the two have very
different costs.

The "Key metrics" block at the end prints the numbers meant for comparing
hardware: the actor update time, the inference wait, the environment step time
without resets (from the first rollout, in which no episode ends in the first
interactions) and the window total.

Usage:
    python scripts/summarize_sac_timing.py train.log [--window 160]
    python scripts/summarize_sac_timing.py train.log --export-csv out_dir [--label run1]

``--export-csv`` writes three files for plotting: ``<label>_steps.csv`` (every
timed step, all ``timing_s/*`` metrics), ``<label>_rollouts.csv`` (one row per
rollout step with its timers and the exact episode counts) and
``<label>_interactions.csv`` (the duration of every interaction of every
rollout and how many episodes finished in it, recovered from the rollout
progress bar; the bar follows stage 0 only).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"step:(\d+) - ")
METRIC = re.compile(r"((?:timing_s|throughput|count)/\w+):([0-9.eE+-]+)")

PROGRESS = re.compile(
    r"Rollout Progress:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)<[^,]*,\s*[0-9.]+s/it,"
    r"\s*done_eps=(\d+), succ_eps=(\d+)"
)

COLLECTED = re.compile(r"Rollout collected trajectories: success=(\d+), failed=(\d+), total=(\d+)")

ROLLOUT_KEYS = (
    "timing_s/step",
    "timing_s/rollout",
    "timing_s/generate",
    "timing_s/update_actor",
    "timing_s/env_loop_total",
    "timing_s/env_loop_env_wait_avg",
    "timing_s/env_loop_rollout_wait_avg",
    "timing_s/env_loop_stage_wall_max",
)


def load_steps(path: str) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    with open(path, errors="ignore") as f:
        for line in f:
            line = ANSI.sub("", line)
            match = STEP.search(line)
            if match is None:
                continue
            metrics = {key: float(value) for key, value in METRIC.findall(line)}
            if "timing_s/step" in metrics:
                steps[int(match.group(1))] = metrics
    return steps


def load_collected(path: str) -> list[tuple[int, int, int]]:
    """Exact (success, failed, total) episode counts printed once per rollout."""
    counts = []
    with open(path, errors="ignore") as f:
        for line in f:
            match = COLLECTED.search(ANSI.sub("", line))
            if match:
                counts.append(tuple(int(g) for g in match.groups()))
    return counts


def load_interactions(path: str) -> list[dict[str, float]]:
    """Per-interaction rows from the rollout progress bar (stage 0 time, episodes of both stages)."""
    rows: list[dict[str, float]] = []
    rollout = -1
    prev_elapsed = prev_done = prev_succ = 0
    last_n = 0
    with open(path, errors="ignore") as f:
        for line in f:
            for m in PROGRESS.finditer(ANSI.sub("", line)):
                n = int(m.group(1))
                elapsed = int(m.group(3) or 0) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
                done, succ = int(m.group(6)), int(m.group(7))
                if n == 0:
                    continue
                if n == last_n:
                    # a repeated refresh of the same interaction (tqdm also prints a final line)
                    continue
                if n < last_n:
                    # a new rollout started: restart the running totals
                    rollout += 1
                    prev_elapsed = prev_done = prev_succ = 0
                elif rollout < 0:
                    rollout = 0
                last_n = n
                rows.append(
                    {
                        "rollout_index": rollout,
                        "interaction": n,
                        "elapsed_s": elapsed,
                        "interaction_s": elapsed - prev_elapsed,
                        "finished_episodes": done - prev_done,
                        "successful_episodes": succ - prev_succ,
                        "finished_episodes_cumulative": done,
                        "successful_episodes_cumulative": succ,
                    }
                )
                prev_elapsed, prev_done, prev_succ = elapsed, done, succ
    return rows


def write_csvs(steps, interactions, collected, out_dir: str, label: str, window: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    keys = sorted({k for m in steps.values() for k in m})
    columns = ["step", "window"] + [
        "step_total" if k == "timing_s/step" else k.replace("timing_s/", "").replace("/", "_") for k in keys
    ]
    path = os.path.join(out_dir, f"{label}_steps.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for s in sorted(steps):
            writer.writerow([s, (s - 1) // window] + [steps[s].get(k, "") for k in keys])
    print(f"wrote {path}")
    rollout_steps = sorted(s for s, m in steps.items() if "timing_s/rollout" in m)
    path = os.path.join(out_dir, f"{label}_rollouts.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["rollout_index", "step", "window"] + [
            k.replace("timing_s/", "").replace("step", "step_total") for k in ROLLOUT_KEYS
        ]
        writer.writerow(header + ["episodes_success", "episodes_failed", "episodes_total"])
        for i, s in enumerate(rollout_steps):
            counts = collected[i] if len(collected) == len(rollout_steps) else ("", "", "")
            row = [i, s, (s - 1) // window] + [steps[s].get(k, "") for k in ROLLOUT_KEYS]
            writer.writerow(row + list(counts))
    print(f"wrote {path}")
    if interactions:
        path = os.path.join(out_dir, f"{label}_interactions.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(interactions[0]))
            writer.writeheader()
            writer.writerows(interactions)
        print(f"wrote {path}")


def print_key_metrics(steps, interactions, collected, window: int) -> None:
    """Numbers that can be compared between hardware configurations."""
    print("\nKey metrics (use window 1 and later; window 0 is shorter because it has fewer resets):")
    rollout_steps = sorted(s for s, m in steps.items() if "timing_s/rollout" in m and (s - 1) % window == 0)
    header = ("window", "total_s", "env_step_s", "inference_s", "first_update_s", "update_s", "episodes")
    print("  " + " ".join(f"{h:>14}" for h in header))
    for i, first in enumerate(rollout_steps):
        w = (first - 1) // window
        in_window = [s for s in steps if (s - 1) // window == w]
        complete = len(in_window) == window
        updates = [steps[s]["timing_s/update_actor"] for s in in_window if s != first and "timing_s/update_actor" in steps[s]]
        row = [
            str(w),
            f"{sum(steps[s]['timing_s/step'] for s in in_window):.0f}" if complete else "incomplete",
            f"{steps[first].get('timing_s/env_loop_env_wait_avg', float('nan')):.1f}",
            f"{steps[first].get('timing_s/env_loop_rollout_wait_avg', float('nan')):.2f}",
            f"{steps[first].get('timing_s/update_actor', float('nan')):.1f}",
            f"{statistics.mean(updates):.3f}" if updates else "-",
            str(collected[i][2]) if i < len(collected) and len(collected) == len(rollout_steps) else "-",
        ]
        print("  " + " ".join(f"{v:>14}" for v in row))
    quiet = []
    for r in (r for r in interactions if r["rollout_index"] == 0):
        if r["finished_episodes"] > 0:
            break
        if r["interaction"] >= 5:
            quiet.append(r)
    if quiet:
        values = [r["interaction_s"] for r in quiet]
        print(
            f"  env step without resets (first rollout, interactions {quiet[0]['interaction']}-{quiet[-1]['interaction']}"
            f" with no finished episode): mean {statistics.mean(values):.1f}s of {values}"
        )
    else:
        print("  env step without resets: no interaction of the first rollout from the 5th on is free of resets")


def describe(values: list[float]) -> str:
    ordered = sorted(values)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    return (
        f"n={len(values)} mean={statistics.mean(values):.3f}s median={statistics.median(values):.3f}s "
        f"std={statistics.pstdev(values):.3f}s p95={p95:.3f}s max={ordered[-1]:.3f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log")
    parser.add_argument("--window", type=int, default=160, help="training steps per rollout window")
    parser.add_argument("--export-csv", metavar="DIR", help="also write per-step and per-interaction CSV files")
    parser.add_argument("--label", help="file name prefix for the CSV files (default: log file name)")
    args = parser.parse_args()

    steps = load_steps(args.log)
    if not steps:
        print("no timed steps found", file=sys.stderr)
        return 1

    rollout_steps = sorted(s for s, m in steps.items() if "timing_s/rollout" in m and (s - 1) % args.window == 0)
    plain_steps = [s for s in steps if s not in rollout_steps]
    print(f"{len(steps)} timed steps, last step {max(steps)}")

    print("\nRollout steps (first step of each window; rollout + one update):")
    for s in rollout_steps:
        row = {key.split("/")[1]: round(steps[s][key], 1) for key in ROLLOUT_KEYS if key in steps[s]}
        print(f"  step {s} (window {(s - 1) // args.window}): {row}")

    updates = [steps[s]["timing_s/update_actor"] for s in plain_steps if "timing_s/update_actor" in steps[s]]
    if updates:
        print("\nPlain update-only steps, update_actor:")
        print(f"  all windows: {describe(updates)}")
        by_window: dict[int, list[float]] = {}
        for s in plain_steps:
            if "timing_s/update_actor" in steps[s]:
                by_window.setdefault((s - 1) // args.window, []).append(steps[s]["timing_s/update_actor"])
        for window, values in sorted(by_window.items()):
            print(f"  window {window}: {describe(values)}")

    interactions = load_interactions(args.log)
    collected = load_collected(args.log)
    print_key_metrics(steps, interactions, collected, args.window)

    if args.export_csv:
        label = args.label or os.path.splitext(os.path.basename(args.log))[0]
        write_csvs(steps, interactions, collected, args.export_csv, label, args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
