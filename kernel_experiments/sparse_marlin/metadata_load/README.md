# Paired 96-bit metadata load experiment

압축을 유지하면서 load를 한 명령으로 줄인 후속 배치는 [OVERLAP.md](OVERLAP.md)에 있습니다. 두 half-warp의 lane들을 묶고 두 행렬을 겹쳐 `ldmatrix.x2`로 읽습니다.

처리량 제한 원인을 분리한 후속 실험은 [BOTTLENECK.md](BOTTLENECK.md)에 있습니다. Scalar shared load의 명령 처리량이 충돌 제거 후의 병목임을 load 수 변경과 `LDS.64` 대조군으로 확인했습니다.

2026-10-19, RTX 3050 4GB Laptop GPU (sm_86, 20 SM), CUDA 12.6, NCU 2024.3.2.

## Question and layout

Compare Sparse-Marlin's `ldmatrix.m8n8.x2` metadata load (two 32-bit registers per thread) against loading the same three 32-bit words into each adjacent pair of threads. A pair's original 32 four-bit entries occupy 128 bits; storing the same entries as 32 three-bit codes takes 96 bits. Even lanes decode bits 0..47; odd lanes decode bits 48..95. Both produce their own original two-register fragment. The mapping is `[4,4,9,8,E,E,D,C]` from the existing encoding experiment.

The compressed input is **already packed for the destination lane pair**, not a drop-in replacement for upstream global packing. Each pair occupies three consecutive words, with no padding. Two identical registers in different threads are still separate physical registers.

## Variants

- `ldmatrix_x2`: original shared layout and one `ldmatrix.x2` instruction.
- `pair_3word_raw`: each thread reads the pair's three words; consumes all words in a checksum, without expansion. Isolates load/address costs but is **not semantically equivalent MMA metadata**.
- `pair_3word_decode`: same three loads, lane-half selection, then two 24-bit → 32-bit SWAR decoders. Produces exactly the original fragments.
- `own_2word_decode`: reads only the two words covering that lane's 48 bits, then decodes. A useful comparison, but this particular interleaved layout introduces bank conflicts.
- `own_2word_raw` (mode 4): reads `(0,1), (1,2), (3,4), (4,5), ...` for lanes 0,1,2,3,..., and returns the two raw words directly. No alignment, bit extraction or decoding.
- `planar_2word_raw` (mode 5): same raw word values as mode 4, but the three words of each pair are stored in three 16-word planes. Eliminates bank conflicts without padding.
- `planar_1load_control` (mode 6): one conflict-free scalar load, returned in both checksum inputs; instruction-throughput control, not full two-word data.
- `planar_4load_control` (mode 7): two repetitions of the two conflict-free scalar loads, with both pairs consumed; instruction-throughput control.
- `padded_64bit_raw` (mode 8): duplicates the middle word in storage so each lane's two raw words are adjacent and 8-byte aligned. Uses one `LDS.64` instruction. Same raw output as mode 5, but the warp tile occupies 256 rather than 192 bytes.
- `overlap_ldmatrix_raw` (mode 9): pair lanes p,p+16 and read the 48-word planar layout using overlapping 32-word matrices with one `ldmatrix.x2`. No padding or shuffles.
- `overlap_ldmatrix_decode` (mode 10): the same load followed by reconstruction of each lane's original two-register 4-bit metadata fragment.

`ld.volatile.shared.b32` ensures repeated scalar loads are not hoisted by ptxas. `asm volatile` alone was insufficient in a discarded preliminary run. The NCU load counts below verify the final executable.

## Measurement

All variants reserve the same 8 KiB static shared memory and prefill four stages before timing. No global loads occur in the repeated loop. Thus GMEM savings, MMA overlap and the real kernel's register pressure are excluded.

1. Dependent mode: one warp, one block, 4,096 iterations. Loaded lane-0 data determines the next shared stage through a shuffle. Includes address generation, reduction checksum, shuffle and loop overhead; these are **not isolated instruction latencies**. Nine samples in randomized order.
2. Throughput mode: 80 blocks × 256 threads × 4,096 iterations. Four shared stages, independent addresses, four-way loop unroll. Nine randomized timing samples. NCU profiles one invocation separately.

Dependent-mode median cycles per iteration:

| Variant | Cycles | Registers/thread |
|---|---:|---:|
| ldmatrix_x2 | 69.26 | 21 |
| pair_3word_raw | 76.26 | 26 |
| pair_3word_decode | 142.27 | 32 |
| own_2word_decode | 138.02 | 27 |

NCU throughput-mode counters (2,621,440 warp iterations):

| Variant | LDSM instructions | Scalar shared load instructions | Shared wavefronts, including initialization | Excess wavefronts | Registers/thread |
|---|---:|---:|---:|---:|---:|
| ldmatrix_x2 | 2,621,440 | 0 | 5,248,000 | 0 | 22 |
| pair_3word_raw | 0 | 7,864,320 | 7,868,160 | 0 | 37 |
| pair_3word_decode | 0 | 7,864,320 | 7,868,160 | 0 | 38 |
| own_2word_decode | 0 | 5,242,880 | 10,489,600 | 5,242,880 | 27 |

Subtracting the initial shared stores, steady-state wavefronts per iteration are respectively 2, 3, 3 and 4. Paired scalar loads have no excess wavefronts: accessing the same word is broadcast, and the 16 distinct addresses of each load hit different banks. Nevertheless, three half-used-bank scalar requests need more wavefronts than the original wide matrix load. Register counts include compiler temporaries and unrolling, not just the metadata payload. No local load/store spills were observed in these microkernels.

NCU total SM cycles were 267,787.5 / 793,838.5 / 1,758,418 / 1,729,980 for modes 0..3. The paired raw and full-decode loops cost about 2.96× and 6.57× the baseline in this throughput test. Wall times vary with unlocked clocks; use counter totals and dependent cycles to interpret the mechanism, not a universal slowdown factor.

The baseline and both full-decode variants match expected output for all 4 stages × 32 lanes × 16 codes (2,048 codes per variant). Compute Sanitizer memcheck and synccheck both report zero errors. This is deterministic random coverage, not exhaustive correctness coverage.

## Real kernel reference

Also profiled the unmodified local Sparse-Marlin kernel at M=1, N=K=4096, group size 128. The selected launch had duration 84.544 us, NCU DRAM throughput 87.01% of peak, 162 registers/thread and achieved occupancy 18.24%. Cache control was `all`, so interpret this as a cold-cache replay measurement. It shows potential value in global compression for this shape; it does not prove metadata alone is the bottleneck. That kernel loads A, B, scales and metadata, and performs reduction and MMA too.

The JSON `mean_us` emitted while NCU is attached includes replay overhead and **must not be used for runtime comparisons**. Use `gpu__time_duration.sum` from the NCU report instead. No compressed metadata variant has been integrated into the real kernel yet, so end-to-end speedup/slowdown is unmeasured.

## Reproduce

```bash
make all sass
make run
make check
make profile
make analyze
```

Raw timing, counter CSVs, SASS and sanitizer logs are under `results/`. Binary NCU reports are generated locally and ignored by Git.

Actual kernel profiling, from the repository root:

```bash
ncu --clock-control none --cache-control all \
  --kernel-name-base function --kernel-name regex:Marlin_2_4 \
  --launch-skip 21 --launch-count 1 --set detailed \
  -o kernel_experiments/sparse_marlin/metadata_load/results/marlin_m1 \
  kernel_bench/.venv/bin/python kernel_experiments/sparse_marlin/run.py bench \
  --m 1 --n 4096 --k 4096 --warmup 20 --iters 3
```

References: [NVIDIA shared memory and broadcast](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks).

## Follow-up: two overlapping word loads without decoding

The requested access is implemented as:

```cpp
const uint32_t* p = stage + 3 * (lane / 2) + (lane & 1);
return make_uint2(shared_load(p), shared_load(p + 1));
```

The latest five-mode run is saved separately in `results/own2_raw/`; the original four-mode measurement above is preserved. Nine randomized samples per mode and kind, 90 samples total. Raw mode 4 was checked against CPU-selected packed words for all 4 stages × 32 lanes; memcheck and synccheck reported zero errors. This verifies the raw word selection, not MMA-compatible output.

| Path | Dependent cycles/iteration | NCU throughput total SM cycles | Relative SM cycles | Wavefronts/iteration | Throughput registers/thread |
|---|---:|---:|---:|---:|---:|
| ldmatrix_x2 | 69.264 | 268,290.5 | 1.000 | 2 | 22 |
| pair_3word_raw | 76.264 | 792,556 | 2.954 | 3 | 37 |
| own_2word_raw | 64.515 | 530,457.5 | 1.977 | 4 | 22 |

The two-word raw path reduces dependent-loop cycles by about 6.9% against ldmatrix, but requires about 1.98× the total SM cycles in the many-warp test. Against the three-word raw path, it reduces throughput-test SM cycles by 33.1%. Neither single-warp latency nor wavefront count alone predicts the aggregate result.

The raw two-word path has exactly 5,242,880 scalar shared load instructions (two per warp iteration) and 5,242,880 excess shared wavefronts. The initial stores account for 3,840 additional wavefronts outside the loop. No local load/store spills were observed. The throughput kernel uses the same 22 registers/thread as the baseline, so extra register allocation does not explain its lower throughput here.

Why bank conflicts occur: the first load accesses word indices `0,1,3,4,...,33,34,...`. Lane 1 reads word 1, while lane 22 reads word 33. These are different words in the same bank (`index % 32 == 1`). The second load has a similar conflict. Each scalar instruction needs two wavefronts, for four in total. Adjacent-lane overlap still broadcasts the shared middle word, but cannot eliminate conflicts with more distant lanes.

The original three-word-pair path instead accesses `3*pair + j`, for fixed j=0,1,2. Its 16 distinct addresses map to distinct banks per instruction, producing three conflict-free wavefronts. Despite having fewer wavefronts, that implementation has more load instructions and different compiler scheduling/register allocation, and measured slower than the two-word path.

Clocks were unlocked. The raw NCU durations were 238.208 / 464.416 / 311.616 us for modes 0 / 1 / 4, at different SM clocks; the table therefore compares SM cycle counters rather than those durations. These are isolated shared-load loops, not end-to-end Sparse-Marlin measurements.

Reproduce the follow-up without replacing the original result directory:

```bash
make all sass RESULTS=results/own2_raw
make run check profile analyze RESULTS=results/own2_raw
```

## Follow-up: change the shared layout

For pair p, name its three original words X[p], Y[p], Z[p]. Change the 48-word warp tile from:

```text
X0 Y0 Z0 X1 Y1 Z1 ... X15 Y15 Z15
```

to:

```text
X0 X1 ... X15 | Y0 Y1 ... Y15 | Z0 Z1 ... Z15
```

This preserves the 192-byte size, with no padding. The source is prepacked into this layout on CPU, so shared initialization copies an already planar input; GPU layout conversion cost is not included.

```cpp
int index = lane / 2 + 16 * (lane & 1);
return make_uint2(shared_load(stage + index), shared_load(stage + index + 16));
```

Even lane 2p receives X[p],Y[p]; odd lane 2p+1 receives Y[p],Z[p], exactly as in mode 4. In load 0, the even lanes access banks 0..15 and the odd lanes banks 16..31. In load 1 these bank ranges swap. Stage base offsets rotate the banks but preserve uniqueness. The entire tile remains 25% smaller than the original 256-byte metadata tile.

All six timing modes were rerun, with 9 randomized samples each for dependent and throughput loops (108 samples). Modes 0,1,4,5 were profiled by NCU. Results are saved separately in `results/planar/`.

| Path | Dependent cycles/iteration | NCU total SM cycles | Shared wavefronts/iteration | Excess wavefronts/iteration |
|---|---:|---:|---:|---:|
| ldmatrix_x2 | 69.264 | 267,983.5 | 2 | 0 |
| own_2word_raw, interleaved | 64.515 | 529,858 | 4 | 2 |
| planar_2word_raw | 62.514 | 529,676 | 2 | 0 |

NCU observed exactly 5,242,880 scalar shared load instructions for each of the two raw paths (two per warp iteration). Including 3,840 initialization stores, total shared wavefronts fall from 10,489,600 to 5,246,720. Excess wavefronts fall from 5,242,880 to zero. Both paths use 22 registers/thread in the throughput kernel, have identical total dynamic instruction counts, and show no local load/store spills.

The layout successfully removes the measured bank conflicts. Dependent-loop cycles fall about 3.1%, but the many-warp total cycle count changes by less than 0.1%; this isolated loop is not throughput-limited by those extra bank-conflict wavefronts. The planar path still costs about 1.98× the ldmatrix baseline's total SM cycles in this test. It issues two scalar load instructions rather than one matrix load, but these measurements do not isolate every remaining scheduling/dependency cost.

The planar and interleaved raw outputs were checked against the same CPU reference for 4 stages × 32 lanes. Memcheck and synccheck report zero errors. This experiment excludes decoding, MMA, global transfer savings and any GPU transpose; end-to-end integration requires corresponding global packing/shared-copy changes.

```bash
make all sass RESULTS=results/planar
make run check profile analyze RESULTS=results/planar MODES="0 1 4 5"
```
