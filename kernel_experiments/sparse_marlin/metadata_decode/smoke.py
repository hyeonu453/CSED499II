"""Real packed weights: compare baseline, fetch decode and matmul decode."""
import argparse,csv,json,random,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'src'))
import torch
import marlin
import metadata_decode as ext
from test_2_4 import gen_quant4_NT

def main():
 p=argparse.ArgumentParser();p.add_argument('--batches',type=int,nargs='+',default=[1,8,16,32,64,128]);p.add_argument('--samples',type=int,default=7);p.add_argument('--calls',type=int,default=40);p.add_argument('--replays',type=int,default=3);p.add_argument('--check-only',action='store_true');p.add_argument('--profile',choices=['baseline','fetch','matmul']);p.add_argument('--seed',type=int,default=20261019);p.add_argument('--checks',type=int,default=3);p.add_argument('--label',default='');args=p.parse_args()
 result=ROOT/'results';result.mkdir(exist_ok=True)
 torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False
 n=k=4096
 ref_b,b,s,meta=gen_quant4_NT(n,k,128)
 packed=ext.pack(meta);restored=ext.unpack(packed)
 torch.cuda.synchronize();assert torch.equal(meta.flatten(),restored),'codec roundtrip mismatch'
 functions={'baseline':marlin.mul_2_4,'fetch':ext.fetch,'matmul':ext.matmul}
 print('GPU:',torch.cuda.get_device_name(),'; metadata bytes:',meta.nbytes,'->',packed.nbytes,flush=True)
 print('PASS: complete real metadata roundtrip bit-exact',flush=True)
 # FP32 oracle for the exact sparse, quantized weights (not original dense weights).
 ref_b32=ref_b.float();checks=[];timings=[]
 for m in args.batches:
  a=torch.randn((m,k),device='cuda',dtype=torch.float16)
  oracle=a.float()@ref_b32
  buffers={v:torch.empty((m,n),device='cuda',dtype=torch.float16) for v in functions}
  work={v:torch.zeros(n//128*16,device='cuda',dtype=torch.int32) for v in functions}
  def invoke(v):functions[v](a,b,meta if v=='baseline' else packed,buffers[v],s,work[v])
  outputs={};drift={};all_trials={}
  for v in functions:
   trials=[]
   for _ in range(args.checks):invoke(v);torch.cuda.synchronize();trials.append(buffers[v].clone())
   all_trials[v]=trials;outputs[v]=trials[-1];drift[v]=max((t.float()-trials[0].float()).abs().max().item() for t in trials)
  base=outputs['baseline'].float()
  for v in functions:
   y=outputs[v].float();err=(y-oracle).abs();diff=(y-base).abs()
   worst=max(((t.float()-oracle).abs().mean()/oracle.abs().mean()).item() for t in all_trials[v]);finite=all(bool(t.isfinite().all()) for t in all_trials[v])
   r=dict(m=m,variant=v,worst_relative_mae=worst,stable=drift[v]==0,passed=finite and worst<0.002 and drift[v]==0,relative_mae=(err.mean()/oracle.abs().mean()).item(),max_abs_error=err.max().item(),vs_baseline_relative_mae=(diff.mean()/base.abs().mean()).item(),vs_baseline_max_abs=diff.max().item(),vs_baseline_equal_pct=(y==base).float().mean().item()*100,repeat_max_abs=drift[v],finite=bool(y.isfinite().all()),pass_relative_mae=bool((err.mean()/oracle.abs().mean())<0.002))
   checks.append(r)
  print('Accuracy',m,[(r['variant'],round(r['relative_mae'],7),r['vs_baseline_max_abs']) for r in checks[-3:]],flush=True)
  if args.profile:
   for _ in range(20):invoke(args.profile)
   torch.cuda.synchronize();torch.cuda.profiler.start();invoke(args.profile);torch.cuda.synchronize();torch.cuda.profiler.stop();break
  if args.check_only:continue
  graphs={}
  for v in functions:
   for _ in range(5):invoke(v)
   torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
    for _ in range(args.calls):invoke(v)
   graphs[v]=g
  jobs=[(v,i) for v in functions for i in range(args.samples)];random.Random(args.seed+m).shuffle(jobs)
  for v,i in jobs:
   graphs[v].replay();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
   for _ in range(args.replays):graphs[v].replay()
   end.record();end.synchronize();timings.append(dict(m=m,variant=v,sample=i,us=start.elapsed_time(end)*1000/(args.calls*args.replays)))
  print('Timing us',m,{v:round(statistics.median(x['us'] for x in timings if x['m']==m and x['variant']==v),3) for v in functions},flush=True)
 if args.profile:return
 suffix=('_check' if args.check_only else '')+('_'+args.label if args.label else '')
 for name,rows in [('accuracy',checks),('timing',timings)]:
  if rows:
   with (result/(name+suffix+'.csv')).open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 if timings:
  summary=[]
  for m in args.batches:
   med={v:statistics.median(x['us'] for x in timings if x['m']==m and x['variant']==v) for v in functions}
   for v in functions:summary.append(dict(m=m,variant=v,median_us=med[v],speedup_vs_baseline=med['baseline']/med[v]))
  with (result/'summary.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 (result/('environment'+suffix+'.json')).write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,k=k,n=n,original_metadata_bytes=meta.nbytes,packed_metadata_bytes=packed.nbytes,**vars(args)),indent=2)+'\n')
if __name__=='__main__':
 with torch.inference_mode():main()
