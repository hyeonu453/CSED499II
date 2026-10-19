"""Summarize timings and register-data dependency DAGs from the real SASS."""
import csv
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
from statistics import median

ROOT=Path(__file__).resolve().parent/"results"
groups=defaultdict(list)
for r in csv.DictReader((ROOT/"timing.csv").open()):groups[r["kind"],r["decoder"],int(r["ilp"])].append(r)
with (ROOT/"summary.csv").open("w") as f:
    w=csv.writer(f)
    w.writerow(["kind","decoder","ilp","samples","median_cycles_per_call","median_us","registers","local_bytes"])
    for key,rows in sorted(groups.items()):
        w.writerow([*key,len(rows),median(float(r["cycles_per_call"]) for r in rows),
                    median(float(r["us"]) for r in rows),rows[0]["registers"],rows[0]["local_bytes"]])

sass=(ROOT/"sass.txt").read_text();dags={}
for d in [0,1,2,4,5]:
    body=sass.split(f"Function : _Z5probeILi{d}EEvPKjPj",1)[1].split("Function :",1)[0]
    registers={};nodes=[]
    for address,line in re.findall(r"/\*([0-9a-f]+)\*/\s+([^;]+);",body):
        op=line.split()[0]
        regs=[int(x) for x in re.findall(r"\bR(\d+)\b",line)]
        if not regs:continue
        if op.startswith("LDG"):
            registers[regs[0]]=(0,[]);continue
        if op.startswith("STG"):
            dest=int(re.search(r"\bR(\d+)\b",line.split("]",1)[1])[1])
            depth,path=registers[dest]
            dags[d]=dict(dependent_instructions=len(nodes),input_to_output_depth=depth,
                         critical_path=path,opcodes=dict(Counter(n["opcode"] for n in nodes)),nodes=nodes)
            break
        dst,*src=regs
        deps=[registers[x] for x in src if x in registers]
        if deps:
            depth,path=max(deps,key=lambda x:x[0])
            node=dict(address=address,opcode=op,instruction=line,depth=depth+1)
            nodes.append(node);registers[dst]=(depth+1,path+[f"{address}:{op}"])
        else:registers.pop(dst,None)
(ROOT/"dependency_dags.json").write_text(json.dumps(dags,indent=2)+"\n")

for d in [0,4,5]:
    r=list(csv.DictReader((ROOT/f"d{d}_ilp8_raw.csv").open()))[-1]
    assert float(r["smsp__sass_inst_executed_op_local_ld.sum"])==0
    assert float(r["smsp__sass_inst_executed_op_local_st.sum"])==0
full=list(csv.DictReader((ROOT/"d0_ilp8_raw.csv").open()))[-1]
fused=list(csv.DictReader((ROOT/"d4_ilp8_raw.csv").open()))[-1]
calls=160*4*256*16*8
delta=float(full["smsp__inst_executed.sum"])-float(fused["smsp__inst_executed.sum"])
# Setup/teardown also differ by a few instructions per warp.
assert abs(delta-2*calls)<=16*160*4
print("Wrote timing summary and SASS dependency DAGs; two fewer steady-state instructions/call (plus setup differences) and zero spills verified.")
