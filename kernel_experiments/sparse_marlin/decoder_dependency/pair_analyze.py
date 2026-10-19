"""Summarize latency separately from SM throughput for two decoded words."""
import csv,statistics
from pathlib import Path
root=Path(__file__).resolve().parent/'results'
rows=list(csv.DictReader((root/'pair_timing.csv').open()))
summary=[]
for d in range(5):
 for layout in range(3):
  for pairs in (1,2,4,8):
   r=[x for x in rows if int(x['decoder'])==d and int(x['layout'])==layout and int(x['pairs'])==pairs and x['kind']=='single_warp']
   assert len(r)==7 and all(int(x['local_bytes'])==0 for x in r)
   summary.append(dict(decoder=d,layout=layout,pairs=pairs,cycles_per_pair=statistics.median(float(x['cycles_per_pair']) for x in r),registers=r[0]['registers']))
with (root/'pair_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print('Mixed lanes, median cycles per 48-bit pair (two output words):')
for d in range(5):print(d,[(x['pairs'],round(x['cycles_per_pair'],3)) for x in summary if x['decoder']==d and x['layout']==2])
print('Saturated NCU cycles per warp-pair per SM:')
for r in csv.DictReader((root/'pair_ncu.csv').open()):
 assert float(r['smsp__sass_inst_executed_op_local_ld.sum'])==float(r['smsp__sass_inst_executed_op_local_st.sum'])==0
 calls=160*4*4096*int(r['pairs'])
 print(r['decoder'],r['pairs'],round(float(r['sm__cycles_elapsed.avg'])*20/calls,3),'ALU',r['sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed'])
