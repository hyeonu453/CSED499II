"""Collect randomized CUDA Graph comparisons without an attached profiler."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
OUT.mkdir(exist_ok=True)
for m,n,k in [(1,4096,4096),(16,4096,4096),(32,4096,4096),(1,1024,1024)]:
    stem=f"m{m}_n{n}_k{k}"
    with (OUT/f"timing_{stem}.csv").open("w") as out, (OUT/f"timing_{stem}.log").open("w") as log:
        subprocess.run([sys.executable,str(ROOT/"run.py"),"bench","--dataset","uniform",
                        "--m",str(m),"--n",str(n),"--k",str(k)],stdout=out,stderr=log,check=True)
    print("Benchmarked",stem,flush=True)
