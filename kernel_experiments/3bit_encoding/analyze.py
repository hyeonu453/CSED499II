#!/usr/bin/env python3
"""Generate reproducible aggregates and auditable SASS opcode inventories."""
import csv, re, json, statistics as st
from collections import defaultdict, Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
rows=list(csv.DictReader((R/'measurements.csv').open()))
groups=defaultdict(list)
for r in rows:groups[r['kind'],r['method'],int(r['block']),int(r['warps'])].append(r)
with (R/'summary.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['kind','method','block','warps','samples','median_ms','median_GBps','min_GBps','max_GBps','median_Gcodes_s','median_cycles','min_cycles','max_cycles'])
 for (kind,m,b,n),rs in sorted(groups.items()):
  vals=lambda key:[float(x[key]) for x in rs]
  w.writerow([kind,m,b,n,len(rs),st.median(vals('ms')),st.median(vals('GBps')),min(vals('GBps')),max(vals('GBps')),st.median(vals('Gcodes_s')),st.median(vals('cycles')),min(vals('cycles')),max(vals('cycles'))])
sass=(R/'sass.txt').read_text();inventory={}
for name,body in re.findall(r'Function : (\S+)(.*?)(?=Function :|\Z)',sass,re.S):
 if 'expansion' not in name and 'probe' not in name:continue
 lines=re.findall(r'/\*([0-9a-f]+)\*/\s+((?:@!?P\d\s+)?[A-Z][^;]*);',body)
 ins=[]
 for addr,line in lines:
  op=re.sub(r'^@!?P\d\s+','',line).split()[0]
  if op=='NOP':continue
  # Trailing self-branch is dead padding, not the executed kernel.
  if op=='BRA' and re.search(r'0x'+addr.lstrip('0')+r'\s*$',line):continue
  ins.append((addr,line,op))
 counts=Counter(x[2] for x in ins)
 # Decoder DAG depth: loaded data roots have depth 0; addresses and predicates
 # are excluded. This is instruction dependency depth, not a cycle prediction.
 depths={};final=[];paths={}
 for addr,line,op in ins:
  clean=re.sub(r'^@!?P\d\s+','',line)
  parts=clean.split(None,1);args=parts[1] if len(parts)>1 else ''
  if op.startswith('LDG'):
   reg=re.search(r'R(\d+)',args)
   if reg:
    base=int(reg[1]);width=4 if '.128' in op else 2 if '.64' in op else 1
    for k in range(width):depths[base+k]=0;paths[base+k]=[f'{addr}:{op}']
   continue
  if op.startswith('STG'):
   regs=re.findall(r'R(\d+)',args.split(']',1)[-1])
   final.extend((depths.get(int(x),0),paths.get(int(x),[])) for x in regs);continue
  if op.startswith(('ISETP','EXIT','BRA','S2R','ULDC')):continue
  regs=[int(x) for x in re.findall(r'\bR(\d+)\b',args)]
  if not regs:continue
  dst,*src=regs
  deps=[(depths[x],paths[x]) for x in src if x in depths]
  if deps:
   dep,path=max(deps,key=lambda x:x[0]);depths[dst]=dep+1;paths[dst]=path+[f'{addr}:{op}']
  else:depths.pop(dst,None);paths.pop(dst,None)
 longest=max(final,default=(0,[]),key=lambda x:x[0])
 inventory[name]={'static_instructions_including_exit_paths':len(ins),'opcodes':dict(counts),'load_to_output_dependency_depth':longest[0],'dependency_path':longest[1]}
 (R/(name+'.sass')).write_text('\n'.join(f'{a}  {l}' for a,l,o in ins)+'\n')
(R/'sass_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
with (R/'profile_summary.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['method_id','metric','value_total','per_warp'])
 for path in sorted(R.glob('ncu_[0-9]*.csv')):
  text=path.read_text();i=text.find('"ID"');
  if i<0:continue
  for r in csv.DictReader(text[i:].splitlines()):
   value=float(r['Metric Value'].replace(',',''));w.writerow([path.stem[4:],r['Metric Name'],value,value/32768])
print('Wrote summary.csv, sass_inventory.json, individual SASS files, profile_summary.csv')
