"""Batch smoke sweep: compare total runtime after removing metadata paths."""
import csv
import json
from pathlib import Path
from statistics import median
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/batch_bottleneck_k4096"
OUT.mkdir(parents=True,exist_ok=True)
all_rows=[];summary=[]
for m in [1,2,4,8,16,24,32,48,64,96,128,192,256]:
    path=OUT/f"batch{m}.csv"
    with path.open("w") as out,(OUT/f"batch{m}.log").open("w") as log:
        subprocess.run([sys.executable,str(ROOT/"run.py"),"bench","--dataset","uniform",
                        "--m",str(m),"--n","4096","--k","4096","--samples","5",
                        "--graph-calls","20","--replays","3","--skip-check"],
                       stdout=out,stderr=log,check=True)
    rows=list(csv.DictReader(path.open()));all_rows.extend(rows)
    times={name:median(float(r["mean_us"]) for r in rows if r["variant"]==name)
           for name in ("baseline","no_gmem","no_reg","no_all")}
    row=dict(batch=m,**times)
    for name in ("no_gmem","no_reg","no_all"):
        row[name+"_reduction_pct"]=100*(1-times[name]/times["baseline"])
    summary.append(row)
    with (OUT/"summary.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(summary)
    print(json.dumps(row),flush=True)
with (OUT/"measurements.csv").open("w") as f:
    w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
