"""Collect NCU counters for dependency and pipeline comparisons."""
import csv
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
keys=["gpu__time_duration.sum","sm__cycles_elapsed.avg","launch__registers_per_thread",
      "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
      "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
      "sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_elapsed",
      "sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_elapsed",
      "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed",
      "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
      "smsp__inst_executed.sum","smsp__pcsamp_sample_count",
      "smsp__pcsamp_warps_issue_stalled_wait","smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
      "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
      "smsp__sass_inst_executed_op_local_ld.sum","smsp__sass_inst_executed_op_local_st.sum"]
summary=[]
for d,ilp in [(0,1),(0,2),(0,8),(4,1),(4,2),(4,8),(5,1),(5,8),(1,8),(2,8)]:
    name=f"d{d}_ilp{ilp}"
    with (OUT/f"{name}.log").open("w") as log:
        subprocess.run(["/usr/local/cuda/bin/ncu","--clock-control","none","--cache-control","all",
            "--set","detailed","--launch-count","1","--kernel-name","decoder_chain",
            "-f","-o",str(OUT/name),str(ROOT/"experiment"),"profile",str(d),str(ilp)],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    raw=OUT/f"{name}_raw.csv"
    with raw.open("w") as out:
        subprocess.run(["/usr/local/cuda/bin/ncu","--import",str(OUT/name)+".ncu-rep",
                        "--csv","--page","raw"],stdout=out,check=True)
    r=list(csv.DictReader(raw.open()))[-1]
    summary.append(dict(decoder=d,ilp=ilp,**{k:r.get(k,"") for k in keys}))
    print("Collected",name,flush=True)
with (OUT/"ncu_summary.csv").open("w") as out:
    w=csv.DictWriter(out,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
