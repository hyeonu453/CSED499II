"""Aggregate original-kernel counters and deduplicated SASS metadata accesses."""
import csv
from collections import Counter, defaultdict
from pathlib import Path
import re
from statistics import median

ROOT=Path(__file__).resolve().parent/"results"
METRICS=["gpu__time_duration.sum","gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
 "dram__bytes_read.sum","lts__throughput.avg.pct_of_peak_sustained_elapsed",
 "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
 "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed",
 "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
 "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
 "sm__warps_active.avg.pct_of_peak_sustained_active","launch__registers_per_thread",
 "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
 "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
 "smsp__pcsamp_warps_issue_stalled_mio_throttle",
 "smsp__pcsamp_warps_issue_stalled_wait","smsp__pcsamp_warps_issue_stalled_barrier",
 "smsp__pcsamp_sample_count","smsp__inst_executed_op_ldsm.sum",
 "smsp__inst_executed_op_ldgsts.sum","smsp__sass_inst_executed_op_shared_st.sum",
 "smsp__sass_inst_executed_op_local_ld.sum","smsp__sass_inst_executed_op_local_st.sum"]


def num(value):
    try:return float(value.replace(",",""))
    except (ValueError,AttributeError):return 0


def sass_rows(path):
    header=None;seen=set()
    for row in csv.reader(path.open()):
        if row and row[0]=="Line No":header=row;continue
        if not header or len(row)!=len(header) or not row[2].startswith("0x") or row[2] in seen:continue
        seen.add(row[2])
        yield dict(zip(header,row))


with (ROOT/"counter_summary.csv").open("w") as out, (ROOT/"metadata_source_summary.csv").open("w") as source:
    writer=csv.writer(out);writer.writerow(["case",*METRICS])
    sw=csv.writer(source)
    sw.writerow(["case","category","warp_instructions","predicated_thread_instructions",
                 "issued_copy_bytes","shared_wavefronts","excess_shared_wavefronts",
                 "stall_long_sb","stall_short_sb","stall_mio","stall_barrier"])
    for path in sorted(ROOT.glob("*_raw.csv")):
        rows=list(csv.DictReader(path.open()))
        if len(rows)<2:continue
        units,row=rows[0],rows[-1]
        name=path.name.removesuffix("_raw.csv")
        vals=[]
        for k in METRICS:
            value=num(row.get(k))
            if k=="gpu__time_duration.sum":value*={"ns":.001,"us":1,"ms":1000,"s":1e6}[units[k]]
            if k=="dram__bytes_read.sum":value*={"byte":1,"Kbyte":1000,"Mbyte":1e6,"Gbyte":1e9}[units[k]]
            vals.append(value)
        writer.writerow([name,*vals])
        src=ROOT/f"{name}_source.csv"
        if not src.exists():continue
        m=int(re.search(r"_m(\d+)_",name)[1])
        bm,bn,bk=(16,128,128) if m<=16 else (min(m,64),256,64)
        meta_base=4*(bm*bk*2+bn*bk//4+bn*2)
        meta_stage=bn*bk//8
        offsets={meta_base+i*meta_stage for i in range(4)}
        totals=defaultdict(Counter)
        for r in sass_rows(src):
            asm=r["Source"];g="other"
            if "LDSM.16.M88.2" in asm:g="metadata_smem_to_reg"
            elif "LDSM.16.M88.4" in asm:g="activation_smem_to_reg"
            elif "LDGSTS" in asm:
                dest=asm.split("[",1)[1].split("]",1)[0]
                off=re.search(r"\+0x([0-9a-f]+)",dest)
                g="metadata_gmem_to_smem" if off and int(off[1],16) in offsets else "other_async_copy"
            elif "HMMA" in asm:g="mma"
            for key in ["Instructions Executed","Predicated-On Thread Instructions Executed",
                        "L1 Wavefronts Shared","L1 Wavefronts Shared Excessive",
                        "stall_long_sb","stall_short_sb","stall_mio","stall_barrier"]:
                totals[g][key]+=num(r.get(key))
        for category,t in totals.items():
            pred=t["Predicated-On Thread Instructions Executed"]
            sw.writerow([name,category,t["Instructions Executed"],pred,
                         pred*16 if category.endswith("copy") or category=="metadata_gmem_to_smem" else 0,
                         t["L1 Wavefronts Shared"],t["L1 Wavefronts Shared Excessive"],
                         t["stall_long_sb"],t["stall_short_sb"],t["stall_mio"],t["stall_barrier"]])

timings=defaultdict(list)
for path in ROOT.glob("timing_*.csv"):
    for r in csv.DictReader(path.open()):
        timings[r["m"],r["n"],r["k"],r["variant"]].append(float(r["mean_us"]))
with (ROOT/"timing_summary.csv").open("w") as out:
    w=csv.writer(out);w.writerow(["m","n","k","variant","samples","median_us","min_us","max_us"])
    for key,vs in sorted(timings.items()):w.writerow([*key,len(vs),median(vs),min(vs),max(vs)])
print("Wrote counter_summary.csv, metadata_source_summary.csv and timing_summary.csv")
