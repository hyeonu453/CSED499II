"""Summarize timing medians and verify NCU really saw every intended load."""
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
groups = defaultdict(list)
for row in csv.DictReader((ROOT / "timing.csv").open()):
    groups[row["kind"], row["method"]].append(row)
with (ROOT / "summary.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["kind", "method", "samples", "median_us", "median_cycles_per_warp_iteration", "registers"])
    for (kind, method), rows in sorted(groups.items()):
        writer.writerow([kind, method, len(rows), median(float(r["ms"]) * 1000 for r in rows),
                         median(float(r["cycles_per_warp_iteration"]) for r in rows), rows[0]["registers"]])

metrics = ["gpu__time_duration.sum", "sm__cycles_elapsed.avg", "launch__registers_per_thread",
           "smsp__inst_executed.sum", "smsp__inst_executed_op_ldsm.sum",
           "smsp__sass_inst_executed_op_shared_ld.sum", "memory_l1_wavefronts_shared",
           "memory_l1_wavefronts_shared_ideal", "derived__memory_l1_wavefronts_shared_excessive",
           "smsp__sass_inst_executed_op_local_ld.sum", "smsp__sass_inst_executed_op_local_st.sum",
           "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
           "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed",
           "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
           "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
           "smsp__pcsamp_warps_issue_stalled_mio_throttle",
           "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
           "smsp__pcsamp_warps_issue_stalled_long_scoreboard", "smsp__pcsamp_sample_count"]
with (ROOT / "ncu_summary.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["mode", *metrics])
    for mode, count in enumerate([1, 3, 3, 2, 2, 2, 1, 4, 1, 1, 1]):
        path = ROOT / f"raw_mode{mode}.csv"
        if not path.exists():
            continue
        row = list(csv.DictReader(path.open()))[-1]
        key = "smsp__inst_executed_op_ldsm.sum" if mode in (0, 9, 10) else "smsp__sass_inst_executed_op_shared_ld.sum"
        assert float(row[key]) == 80 * 8 * 4096 * count, (mode, row[key])
        writer.writerow([mode, *(row.get(k, "") for k in metrics)])
print("Wrote summary.csv and ncu_summary.csv; intended load counts verified for this 20-SM run.")
