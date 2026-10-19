"""Small batch sweep of the unmodified Sparse-Marlin kernel at K=N=4096."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"src"))


def main():
    import numpy as np
    import torch
    import marlin
    # Import first: the upstream test helper resets RNG seeds on import.
    from test_2_4 import gen_quant4_NT
    torch.manual_seed(20261019);np.random.seed(20261019)
    torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/"results/smoke_k4096"
    out.mkdir(parents=True,exist_ok=True)
    n=k=4096
    batches=[1,2,4,8,16,24,32,48,64,96,128,192,256]
    with torch.inference_mode():
        ref_b,b,s,meta=gen_quant4_NT(n,k,128)
        ref32=ref_b.float()
        all_a=torch.randn((max(batches),k),device="cuda",dtype=torch.float16)
        # A common weight and prefix of A allow comparison across batch sizes.
        reference=all_a.float()@ref32
        rows=[]
        for m in batches:
            a=all_a[:m].contiguous()
            c=torch.empty((m,n),device="cuda",dtype=torch.float16)
            workspace=torch.zeros(n//128*16,device="cuda",dtype=torch.int32)

            def invoke():marlin.mul_2_4(a,b,meta,c,s,workspace)

            error=[]
            for _ in range(3):
                invoke();torch.cuda.synchronize()
                error.append(((c.float()-reference[:m]).abs().mean()/reference[:m].abs().mean()).item())
            finite=bool(torch.isfinite(c).all().item())
            for _ in range(5):invoke()
            torch.cuda.synchronize()
            samples=[]
            # Capture launches to keep host launch gaps out of this smoke timing.
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(20):invoke()
            for _ in range(3):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();g.replay();end.record();end.synchronize()
                samples.append(start.elapsed_time(end)*1000/20)
            row=dict(batch=m,n=n,k=k,groupsize=128,relative_mae_min=min(error),
                     relative_mae_max=max(error),finite=finite,
                     passed=finite and max(error)<0.002,median_us=sorted(samples)[1])
            rows.append(row)
            print(json.dumps(row),flush=True)
            del g
        with (out/"results.csv").open("w") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (out/"environment.json").write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,cuda=torch.version.cuda,seed=20261019,
            reference="FP32 matmul, TF32 disabled",threshold=0.002,
            correctness_repeats=3,timing="3 CUDA event samples, 20 graph-captured calls per sample",
            note="Failed cases are timed only for diagnosis; timing is not a valid performance result."),indent=2)+"\n")
        print(f"PASS {sum(r['passed'] for r in rows)}/{len(rows)}; saved {out}",file=sys.stderr)


if __name__=="__main__":main()
