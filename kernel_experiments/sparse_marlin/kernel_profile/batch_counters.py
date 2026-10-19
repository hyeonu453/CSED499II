"""Small NCU follow-up for batch timing smoke; no correctness gate."""
import csv
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/batch_bottleneck_k4096"
OUT.mkdir(parents=True,exist_ok=True)
metrics=["gpu__time_duration.sum","sm__cycles_elapsed.avg",
 "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
 "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
 "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed",
 "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
 "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
 "launch__registers_per_thread","smsp__sass_inst_executed_op_local_ld.sum",
 "smsp__sass_inst_executed_op_local_st.sum"]
summary=[]
for m in [1,32,48,64,256]:
    stem=OUT/f"ncu_batch{m}"
    with Path(str(stem)+".log").open("w") as log:
        subprocess.run(["/usr/local/cuda/bin/ncu","--profile-from-start","off",
            "--launch-count","1","--cache-control","none","--clock-control","none",
            "--metrics",",".join(metrics),"-f","-o",str(stem),sys.executable,
            str(ROOT/"run.py"),"profile","--dataset","uniform","--skip-check","--m",str(m)],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    path=Path(str(stem)+"_raw.csv")
    with path.open("w") as out:
        subprocess.run(["/usr/local/cuda/bin/ncu","--import",str(stem)+".ncu-rep",
                        "--csv","--page","raw"],stdout=out,check=True)
    r=list(csv.DictReader(path.open()))[-1]
    summary.append(dict(batch=m,**{k:r[k] for k in metrics}))
    print("Profiled batch",m,flush=True)
with (OUT/"ncu_summary.csv").open("w") as out:
    w=csv.DictWriter(out,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
