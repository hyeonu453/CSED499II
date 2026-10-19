"""Profile original Sparse-Marlin; run constant-metadata diagnostic ablations."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"src"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["profile","bench"])
    p.add_argument("--variant",choices=["baseline","no_gmem","no_reg","no_all"],default="baseline")
    p.add_argument("--dataset",choices=["packed","uniform"],default="packed")
    p.add_argument("--m",type=int,default=1)
    p.add_argument("--n",type=int,default=4096)
    p.add_argument("--k",type=int,default=4096)
    p.add_argument("--samples",type=int,default=9)
    p.add_argument("--skip-check",action="store_true",help="Time diagnostic ablations even if outputs differ")
    p.add_argument("--graph-calls",type=int,default=100)
    p.add_argument("--replays",type=int,default=5)
    args=p.parse_args()
    if min(args.samples,args.graph_calls,args.replays)<1:p.error("sample and replay counts must be positive")
    if args.action=="bench" and args.dataset!="uniform":
        p.error("Ablations require uniform metadata; choose --dataset uniform")
    if args.variant!="baseline" and args.dataset!="uniform":
        p.error("Ablations are only valid for uniform metadata")
    import numpy as np
    import torch
    import marlin
    if not torch.cuda.is_available():p.exit(1,"CUDA unavailable\n")
    torch.manual_seed(20261019);np.random.seed(20261019)
    m,n,k=args.m,args.n,args.k
    a=torch.randn((m,k),device="cuda",dtype=torch.float16)
    if args.dataset=="packed":
        from test_2_4 import gen_quant4_NT
        ref_b,b,s,meta=gen_quant4_NT(n,k,128)
    else:
        b=torch.randint(-(2**31),2**31-1,(k//32,n*2),device="cuda",dtype=torch.int32)
        s=torch.full((k//128,n),0.01,device="cuda",dtype=torch.float16)
        meta=torch.full((n,k//16),-4370,device="cuda",dtype=torch.int16)
    c=torch.empty((m,n),device="cuda",dtype=torch.float16)
    workspace=torch.zeros(n//128*16,device="cuda",dtype=torch.int32)
    functions={"baseline":marlin.mul_2_4}
    if args.action=="bench" or args.variant!="baseline":
        import metadata_diagnostics
        functions.update({v:getattr(metadata_diagnostics,v) for v in ("no_gmem","no_reg","no_all")})

    def invoke(name):functions[name](a,b,meta,c,s,workspace)

    with torch.inference_mode():
        invoke("baseline");torch.cuda.synchronize()
        reference=c.clone()
        if args.dataset=="packed":
            dense_ref=a@ref_b
            error=((c-dense_ref).abs().mean()/dense_ref.abs().mean()).item()
            assert error<0.002,error
            print(f"baseline relative MAE={error}",file=sys.stderr)
        else:
            for name in functions:
                invoke(name);torch.cuda.synchronize()
                if not args.skip_check:
                    torch.testing.assert_close(c,reference,rtol=0,atol=0)
            if args.skip_check:
                print("Diagnostic timing: output equality checks skipped as requested",file=sys.stderr)
            else:
                print("PASS: all uniform-metadata ablations match baseline bit-exactly",file=sys.stderr)
        if args.action=="profile":
            for _ in range(20):invoke(args.variant)
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            invoke(args.variant)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            print(json.dumps(vars(args)))
            return
        graphs={}
        for name in functions:
            for _ in range(3):invoke(name)
            torch.cuda.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(args.graph_calls):invoke(name)
            graphs[name]=g
        jobs=[(name,i) for name in functions for i in range(args.samples)]
        random.Random(20261019).shuffle(jobs)
        writer=csv.writer(sys.stdout)
        writer.writerow(["m","n","k","dataset","variant","sample","mean_us"])
        for name,i in jobs:
            graphs[name].replay()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.replays):graphs[name].replay()
            end.record();end.synchronize()
            writer.writerow([m,n,k,args.dataset,name,i,start.elapsed_time(end)*1000/(args.graph_calls*args.replays)])
            sys.stdout.flush()


if __name__=="__main__":main()
