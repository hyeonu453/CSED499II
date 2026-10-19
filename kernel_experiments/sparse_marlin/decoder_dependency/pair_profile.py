"""Profile the mixed-lane 48-bit pair path; each launch uses 160 blocks x 128 threads."""
import csv,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'results'
metrics=['sm__cycles_elapsed.avg','gpu__time_duration.sum','launch__registers_per_thread','sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed','sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_elapsed','sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed','smsp__inst_executed.sum','smsp__sass_inst_executed_op_local_ld.sum','smsp__sass_inst_executed_op_local_st.sum']
rows=[]
for d,i in [(0,0),(1,0),(1,3),(2,0),(2,1),(2,3),(4,0),(4,3)]:
 path=out/f'pair_d{d}_i{i}.csv'
 with path.open('w') as f:
  subprocess.run(['/usr/local/cuda/bin/ncu','--clock-control','none','--cache-control','all','--metrics',','.join(metrics),'--csv','--log-file',str(path),str(root/'pair_experiment'),'profile',str(d),str(i),'2'],stdout=subprocess.DEVNULL,check=True)
 lines=path.read_text().splitlines();start=next(j for j,s in enumerate(lines) if s.startswith('"ID"'))
 row={'decoder':d,'pairs':1<<i}
 for x in csv.DictReader(lines[start:]):row[x['Metric Name']]=x['Metric Value'].replace(',','')
 rows.append(row);print('Profiled',d,1<<i,flush=True)
with (out/'pair_ncu.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['decoder','pairs']+metrics);w.writeheader();w.writerows(rows)
