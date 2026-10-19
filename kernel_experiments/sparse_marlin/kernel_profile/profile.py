"""Serial NCU collection; profiler start/stop selects exactly one kernel."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
OUT.mkdir(exist_ok=True)
p=argparse.ArgumentParser()
p.add_argument("phase",choices=["baseline","ablations"])
args=p.parse_args()
jobs=[]
if args.phase=="baseline":
    for m,n,k in [(1,4096,4096),(16,4096,4096),(32,4096,4096),(1,1024,1024)]:
        for cache in ("all","none"):
            jobs.append((m,n,k,"packed","baseline",cache))
else:
    for variant in ("baseline","no_gmem","no_reg","no_all"):
        for cache in ("all","none"):
            jobs.append((1,4096,4096,"uniform",variant,cache))
for m,n,k,dataset,variant,cache in jobs:
    stem=f"{dataset}_{variant}_m{m}_n{n}_k{k}_{cache}"
    if (OUT/f"{stem}_raw.csv").exists() and (OUT/f"{stem}_source.csv").exists():
        print("Already collected",stem,flush=True)
        continue
    report=OUT/stem
    cmd=["/usr/local/cuda/bin/ncu","--clock-control","none","--cache-control",cache,
         "--profile-from-start","off","--launch-count","1","--set","detailed",
         "--import-source","yes","-f","-o",str(report),sys.executable,str(ROOT/"run.py"),
         "profile","--dataset",dataset,"--variant",variant,"--m",str(m),"--n",str(n),"--k",str(k)]
    with (OUT/f"{stem}.log").open("w") as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    for page in ("raw","source"):
        with (OUT/f"{stem}_{page}.csv").open("w") as out:
            export=["/usr/local/cuda/bin/ncu","--import",str(report)+".ncu-rep","--csv","--page",page]
            if page=="source":export += ["--print-source","cuda,sass"]
            subprocess.run(export,stdout=out,check=True)
    print("Collected",stem,flush=True)
