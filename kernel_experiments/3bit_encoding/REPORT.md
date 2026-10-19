# 3b → 4b metadata decoder / 48B → 64B expansion

실측: 2026-09-30, RTX 3050 4GB Laptop (20 SM, sm_86), CUDA 12.6.85, Nsight Compute 2024.3.2, WSL. 소스와 모든 raw 결과를 이 폴더에 보관했다.

## 결과와 추천

측정한 10개 구현 중 대용량 sustained throughput과 warm-cache warp latency 모두 **`direct_u32_4`, 128 threads/block**이 가장 좋았다. 한 warp의 32개 lane이 각자 코드 4개를 처리한다. 각 lane은 인접한 aligned 32-bit word 두 개를 읽고 funnel shift로 12-bit를 추출한 뒤, 네 코드를 동시에 decode하여 16-bit를 저장한다. 입력 48B 형식 변경이나 shuffle이 필요 없다.

- 유효 read+write throughput: **34.71 GB/s**, **39.67 Gcodes/s**; 1,048,576개 warp 작업(입력 48 MiB + 출력 64 MiB).
- 한 warp의 128-code 변환으로 환산하면 aggregate **309.89 Mwarp-jobs/s**다. 역수는 동시 실행 전체의 처리 간격이며 단일 warp latency가 아니다.
- warm-cache 주소 의존 반복: **145.09 SM cycles/warp-job**, 루프·sync·다음 주소 생성 비용 포함.
- registers/thread: **12**, spill/local/shared memory: **0**.
- scalar 반복 baseline 대비 **2.09×**, 1-load + 2-shuffle의 `shuffle_swar4` 대비 **1.057×**.

**Load request와 L1 sector 비용을 우선 최소화하려면 `shuffle_swar4`**가 더 알맞다. LDG 1개, SHFL 2개, 평균 이론적 요청 sector 2개로 정확히 48B를 읽는다. `direct_u32_4`는 LDG 2개, shuffle 0개, 이론적 요청 sector 4개다. 중복 load가 coalescing/cache로 합쳐져 DRAM read는 두 방식 모두 대략 48B/warp였지만, L1 request 비용까지 같은 것은 아니다. 다른 kernel과 합칠 때는 이 차이를 다시 비교해야 한다.

이는 비교한 구현과 현재 장비 상태에서의 최솟값이다. 가능한 모든 bit permutation/producer layout에 대한 전역 최적성 증명은 아니다.

## 정확성 및 encoding

입력 code i는 bit offset 3*i, 출력 metadata i는 bit offset 4*i에 LSB-first로 저장한다. 2-bit 하위 index가 첫 위치, 상위 index가 두 번째 위치다.

| x | y (hex) | y (binary) | nonzero 위치 |
| --- | --- | --- | --- |
| 0 | 0x4 | 0100 | (0, 1) |
| 1 | 0x4 | 0100 | (0, 1) |
| 2 | 0x9 | 1001 | (1, 2) |
| 3 | 0x8 | 1000 | (0, 2) |
| 4 | 0xE | 1110 | (2, 3) |
| 5 | 0xE | 1110 | (2, 3) |
| 6 | 0xD | 1101 | (1, 3) |
| 7 | 0xC | 1100 | (0, 3) |

출력 집합은 `{0x4, 0x8, 0x9, 0xC, 0xD, 0xE}`로 6개 조합 전체와 일치한다. `0/1 → 4`, `4/5 → E`는 alias이므로 이 mapping은 injective하지 않다. Metadata 표현에는 문제가 없으나 원래 3-bit code로의 유일한 역변환은 불가능하다. 위치 index 정의는 [NVIDIA PTX sparse storage 문서](https://docs.nvidia.com/cuda/parallel-thread-execution/#sparse-mma-storage)를 따랐다.

검증 결과:

- GPU scalar probe의 8개 출력을 상수 truth table과 비교: 통과.
- packed 8-code decoder에 대해 **2^24 = 16,777,216개 입력을 GPU에서 전수 검사**: 오류 0. scalar 8회 reference와 비교.
- 10가지 expansion × 4가지 block 크기 각각 **134,217,728개 code**를 CPU reference와 bit-exact 비교: 모두 통과. Seed=20260930, 난수 및 각 code 반복 패턴 포함.
- warp 개수 `1, 2, 3, 7, 9, 31, 33`의 마지막 partial block: 통과.
- 별도 4,099-warp allocation으로 Compute Sanitizer memcheck/synccheck 실행: 모두 **0 errors**. 마지막 warp의 load도 실제 allocation 경계를 넘지 않는다.

## Experiment 1: decoder와 SASS

C++ 식 `3u ^ (x | (7u + (x >> 1)))`의 실제 sm_86 SASS 계산부는 다음 두 명령이다. 주소 계산과 LDG/STG는 제외했다.

```text
LEA.HI   R7, R2, 0x7, RZ, 0x1f
LOP3.LUT R7, R7, 0x3, R2, 0x36, !PT
```

`LEA.HI`가 shift와 add를 합친다. 따라서 PTX에서 SHR/ADD/LOP3 세 개를 적더라도 실제 기계어는 두 개다. 데이터 의존 경로도 `x → LEA.HI → LOP3`의 2단계다. 원본은 `results/_Z5probePKjPj.sass`에 있다.

8개의 code가 nibble 하위 3비트에 이미 배치되어 있다면, 다음 SWAR 식으로 한 번에 decode할 수 있다.

```cpp
uint32_t y = (x | (((x >> 1) & 0x33333333u) + 0x77777777u))
             ^ 0x33333333u;
```

각 nibble에서 shift 후 값은 0..3, 7을 더한 결과는 7..10이므로 nibble 사이 carry가 없다. 실제 SASS는 `SHF → LOP3(mask) → IADD3 → LOP3`의 4명령/4단계다. 단, **연속 3-bit packing을 nibble packing으로 옮기는 비용은 별도**다.

```cpp
// 입력: 하위 24 bits에 8개 3-bit code
x = ((x & 0x00fff000u) << 4) | (x & 0x00000fffu);
x = ((x & 0x0fc00fc0u) << 2) | (x & 0x003f003fu);
x = ((x & 0x38383838u) << 1) | (x & 0x07070707u);
// 위 SWAR 식 적용
```

위 세 spreading 단계는 각각 shift 1개 + LOP3 2개, 총 9개이며 decode 4개까지 합해 **13 instructions/8 codes**다. `packed_probe` SASS로 확인했다. 4-code/12-bit 입력은 첫 spreading 단계가 필요 없어 **6 + 4 = 10 instructions/4 codes**다. 32개 lane 전체를 활용하는 warp expansion에서는 이 더 짧은 사슬이 유리했다.

| 연산 | serial cycles/op | 8 independent chains cycles/op | single-warp codes/cycle |
| --- | --- | --- | --- |
| scalar, feedback mask 없음 | 8.221 | 4.078 | 7.85 |
| 8-nibble SWAR, spreading 제외 | 16.221 | 8.078 | 31.69 |
| scalar + 유효 입력 feedback mask | 12.221 | 6.078 | 5.26 |
| SWAR + 유효 입력 feedback mask | 20.221 | 10.078 | 25.40 |
| IMAD recurrence 참고 | 4.377 | 2.063 | 해당 없음 |

32회 unroll × 1,024 반복, 7회 측정 중앙값이다. cycles/op는 warp-wide invocation 하나당 값이고 SWAR invocation 하나가 lane당 8개 code를 처리한다. 약 0.22 cycles/op의 loop/timer 비용을 포함한다. Raw scalar latency는 8.221 cycles/op, raw SWAR는 16.221 cycles/op로, 각각 2단계·4단계 × 약 4 cycles와 부합한다. Feedback mask는 4 cycles를 추가한다. IMAD control은 별도 연산의 참고치이며 이를 decoder에서 기계적으로 빼지 않았다.

Raw recurrence는 출력을 바로 다음 입력에 넣어 3-bit domain을 벗어날 수 있지만 같은 분기 없는 SASS를 측정한다. 유효 domain 유지 버전도 따로 측정했다. Unrolled SASS에서 실제 반복 개수가 유지됨을 확인했다.

| GPU 전체 포화 연산 | median Gcodes/s | 범위 Gcodes/s |
| --- | --- | --- |
| scalar | 229.64 | 229.49–675.45 |
| 8-nibble SWAR | 603.47 | 534.54–688.77 |

포화 측정은 160 blocks × 128 threads, lane당 8개 independent chains, chain당 8,192회 연산이다. 입력 load/spreading과 최종 expansion store는 포함하지 않는 compute microbenchmark이며, tiny 결과 store와 loop 비용은 포함한다. 서로 다른 실행의 DVFS 상태가 같다고 가정할 수 없으므로 SWAR의 이상적 산술 이득과 실측 Gcodes/s 비율은 구분해야 한다.

## Experiment 2: 구현과 비용

| ID / method | 분담 | load/redistribution |
| --- | --- | --- |
| 0 / shuffle_scalar | 16 lanes × 8 | 12 lanes u32 + 2 shuffles; scalar 8회 |
| 1 / shuffle_swar | 16 lanes × 8 | 동일 load, SWAR 8개 |
| 2 / direct_u32 | 16 lanes × 8 | 각 output lane이 u32 2회 직접 load |
| 3 / direct_u8 | 16 lanes × 8 | 각 output lane이 byte 3개 직접 load |
| 4 / vector_u128 | 16 lanes × 8 | 3 lanes × uint4 load; 8 shuffles + register 선택 |
| 5 / transposed_u8 | 16 lanes × 8 | byte plane을 16B씩 transpose; byte load 3회 |
| 6 / coop_u16 | 16 lanes × 8 | 16 lanes + 8 lanes의 u16 load; 4 shuffles |
| 7 / shuffle_swar4 | 32 lanes × 4 | 12 lanes u32 + 2 shuffles; SWAR 4개 |
| 8 / direct_u8_4 | 32 lanes × 4 | 각 lane byte 2개 load, 0/4-bit shift |
| 9 / direct_u32_4 | 32 lanes × 4 | 각 lane u32 2개 load, funnel shift |

모든 구현은 한 warp마다 정확히 48B 범위 안에서 읽고 64B를 출력한다. 추가 padding/overread는 없다. 마지막 word 다음의 load는 마지막 word로 clamp하고, 필요 없는 상위 bit는 mask한다. Shuffle의 모든 source lane은 참여하며 inactive loader 값은 0으로 초기화했다. Block의 warp들은 독립적이고 shared memory와 block barrier는 없다.

`transposed_u8`만 `p[16*b+j] = original[3*j+b]` 저장 형식이다. Producer가 직접 이 형식을 생성한다고 가정하며 CPU에서 미리 변환했다. 재배치 비용을 시간에 넣지 않았는데도 winner보다 느렸으므로, 별도 GPU transpose를 추가할 근거가 없다.

| method | LDG | SHFL | integer/bitwise 정적 수 | regs/thread | LDG 후 의존 단계 | warm recurrence cycles |
| --- | --- | --- | --- | --- | --- | --- |
| shuffle_scalar | 1 | 2 | 64 | 16 | 15 | 226.08 |
| shuffle_swar | 1 | 2 | 31 | 14 | 15 | 201.08 |
| direct_u32 | 2 | 0 | 32 | 12 | 14 | 176.09 |
| direct_u8 | 3 | 0 | 27 | 12 | 15 | 185.08 |
| vector_u128 | 1 | 8 | 50 | 23 | 16 | 235.09 |
| transposed_u8 | 3 | 0 | 33 | 14 | 15 | 180.08 |
| coop_u16 | 2 | 4 | 45 | 19 | 18 | 216.09 |
| shuffle_swar4 | 1 | 2 | 25 | 14 | 12 | 184.08 |
| direct_u8_4 | 2 | 0 | 29 | 12 | 12 | 148.08 |
| direct_u32_4 | 2 | 0 | 25 | 12 | 11 | 145.09 |

LDG/SHFL/ALU 수는 SASS의 warp instruction 수다. ALU 열은 주소 계산을 포함하며 MOV, predicate/control, LDG/STG/SHFL은 제외한다. Predication 때문에 모든 lane이 각 명령을 실행하는 것은 아니다. `sass_inventory.json`에는 opcode별 전체 수와 주소가 있는 의존 경로를 보관했다.

의존 단계는 load 결과부터 store에 전달되는 decoded 값까지의 **register dataflow DAG 길이**로, 주소 준비와 memory latency를 제외한 정적 분석이다. 독립된 두 load/two shuffle은 서로 직렬 합산하지 않는다. 예를 들어 winner는 `LDG → SHF → spreading 6단계 → decode 4단계`의 load 이후 11단계다. Branch/predicate와 cross-lane schedule까지 고려한 하드웨어 critical-cycle 예측치는 아니다.

Latency는 single warp에서 4,096회 반복하고 output lane 0의 값으로 다음 load 주소를 정한다. `syncwarp` 및 추가 broadcast shuffle, 주소 생성, 루프를 포함하며 마지막 sink store는 timing 밖이다. 9회 중앙값이며 input은 반복 중 L1에 상주한다. 따라서 **cold DRAM latency나 STG completion latency가 아닌 warm-cache load+decode recurrence latency**다. SASS instruction 수는 expansion kernel, latency는 동일 helper를 inline한 별도 kernel에서 측정했다.

### Memory transactions: 측정과 이론 구분

128 threads/block, 32,768 warp jobs, profiler cache-control=all, clock-control=none으로 수집했다. Sector는 32B 단위이며 L1 request, L1 sector access, DRAM sector를 각각 보고한다. 정의는 [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#memory-tables)의 memory workload 분석을 따른다.

| method | lane load bytes | 이론 L1 sectors | 실측 LD requests | 실측 LD sectors | DRAM read sectors | 실행 warp inst | 실행 integer thread-inst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shuffle_scalar | 48 | 2 | 1.00 | 1.940 | 1.513 | 78.0 | 1316.0 |
| shuffle_swar | 48 | 2 | 1.00 | 1.938 | 1.503 | 45.0 | 788.0 |
| direct_u32 | 128 | 4 | 2.00 | 3.937 | 1.509 | 45.0 | 672.0 |
| direct_u8 | 48 | 6 | 3.00 | 5.935 | 1.505 | 41.0 | 592.0 |
| vector_u128 | 48 | 2 | 1.00 | 1.943 | 1.507 | 68.0 | 1087.0 |
| transposed_u8 | 48 | 3 | 3.00 | 3.000 | 1.503 | 46.0 | 672.0 |
| coop_u16 | 48 | 2.5 | 2.00 | 2.500 | 1.511 | 64.0 | 1096.0 |
| shuffle_swar4 | 48 | 2 | 1.00 | 1.935 | 1.503 | 38.0 | 856.0 |
| direct_u8_4 | 64 | 4 | 2.00 | 3.941 | 1.503 | 42.0 | 1088.0 |
| direct_u32_4 | 256 | 4 | 2.00 | 3.941 | 1.503 | 37.0 | 928.0 |

모든 수는 warp-job당이다. `lane load bytes`는 lane별 요청 크기의 합이라 broadcast/중복도 세며 실제 메모리 전송량이 아니다. 연속 48B stride는 32B sector 두 개를 걸치지만 인접 warp끼리 한 sector를 공유하므로 전체 데이터의 unique sector 수는 **1.5/warp**다. DRAM read 실측은 약 1.5 sectors/warp(48B)이고 소량의 부가 트래픽도 포함한다. 중복 word load를 byte 수 그대로 DRAM 트래픽으로 계산하면 안 된다.

이론 L1 sectors는 각 LDG instruction에서 주소가 닿는 unique sector를 합한 값이다. 실제 카운터는 약간 작게 측정되는 경우가 있으므로 둘을 동일시하지 않았다. 그 차이의 구체적인 microarchitecture 원인은 이번 실험에서 분리 검증하지 않았다. Thread-inst는 active predicate를 반영한 thread별 합이며 warp-inst와 다른 단위다.

모든 구현에서 store는 warp당 **1 request / 2 L1 sectors**였다. DRAM write 카운터는 kernel 종료 시 아직 cache에 남은 write 때문에 정확히 2가 아닐 수 있다. `profile_summary.csv`에 store/DRAM read-write/시간/cycle 원본 집계를 모두 포함했다. Cold profiler의 kernel duration은 cache flush와 clock 상태가 다른 별도 실행이므로 아래 sustained 수치와 합치지 않았다.

### Sustained throughput / block size

각 셀은 5회 반복의 중앙값, 단위는 **유효 GB/s = (48+64) × warp_jobs / elapsed_seconds / 10^9**다. 이는 요청된 알고리즘의 useful byte rate이며 실제 DRAM bus bandwidth가 아니다. CUDA event가 output store를 포함한 전체 kernel 실행을 측정한다. Codes/s = GB/s × 128/112.

**Large: 1,048,576 warp jobs, 112 MiB working set (L2 1 MiB보다 큼)**

| method | 32 threads | 64 threads | 128 threads | 256 threads |
| --- | --- | --- | --- | --- |
| shuffle_scalar | 6.55 | 14.66 | 16.62 | 16.46 |
| shuffle_swar | 6.73 | 16.45 | 30.63 | 30.09 |
| direct_u32 | 6.77 | 16.36 | 30.48 | 29.97 |
| direct_u8 | 6.79 | 16.42 | 33.15 | 32.17 |
| vector_u128 | 6.36 | 18.41 | 19.79 | 19.62 |
| transposed_u8 | 6.78 | 16.61 | 29.59 | 28.65 |
| coop_u16 | 6.54 | 18.79 | 22.35 | 21.78 |
| shuffle_swar4 | 6.84 | 16.46 | 32.83 | 31.71 |
| direct_u8_4 | 6.91 | 15.97 | 32.76 | 32.05 |
| direct_u32_4 | 6.98 | 16.29 | 34.71 | 33.93 |

**Small: 4,096 warp jobs, 448 KiB working set (L2에 들어갈 수 있는 크기)**

| method | 32 threads | 64 threads | 128 threads | 256 threads |
| --- | --- | --- | --- | --- |
| shuffle_scalar | 6.37 | 13.31 | 14.20 | 13.99 |
| shuffle_swar | 6.52 | 13.99 | 23.24 | 23.51 |
| direct_u32 | 6.60 | 14.08 | 23.40 | 23.57 |
| direct_u8 | 6.60 | 14.10 | 24.70 | 24.77 |
| vector_u128 | 6.16 | 14.94 | 16.26 | 16.18 |
| transposed_u8 | 6.58 | 14.03 | 22.89 | 22.63 |
| coop_u16 | 6.32 | 14.63 | 17.73 | 17.68 |
| shuffle_swar4 | 6.60 | 14.04 | 24.66 | 24.77 |
| direct_u8_4 | 6.71 | 13.94 | 24.43 | 24.68 |
| direct_u32_4 | 6.82 | 14.09 | 25.72 | 25.89 |

Small working set을 L2-resident라고 단정하지는 않는다. 이 표의 throughput 측정 자체에는 cache hit-rate profiling을 병행하지 않았다. Small에서는 winner의 256-thread 결과가 128-thread보다 약간 높지만 대용량에서는 128이 우세하다. 32/64-thread block의 낮은 값에는 block scheduling 및 residency 제한도 포함된다. 이를 ALU decoder throughput만의 차이로 해석하면 안 된다.

CUDA Graph 한 개에 small 100개 / large 15개 kernel을 넣고, graph warmup 1회 후 3회 replay 시간을 합산했다. 총 400개 설정/trial을 seeded random 순서로 섞었다. H2D/D2H, allocation, graph 생성, CPU packing은 timing 밖이다. 기존 host launch 측정은 참고 파일로만 남겼다.

## 측정의 한계와 적용 범위

Telemetry에서 SM clock이 **210–1125 MHz** 사이로 변했다. 대용량 지속 실행 구간에서는 210 MHz가 관찰되었고, 전력/thermal 상태는 고정하지 않았다. GPU 사용 중 power 약 11–15W, 온도는 약 57–63°C 수준이었다. Property의 nominal 1125 MHz를 모든 clock64 결과에 적용해서 ns로 변환하지 않았다. 근본적인 저클록 원인이 전원 정책인지 power/thermal 제한인지는 판별하지 않았다.

Small throughput 일부 trial은 일시적으로 약 106–111 GB/s까지 올라갔다. 최종 표는 중앙값을 사용했고 원본과 min/max를 `summary.csv`에 보존했다. Large winner는 5회 **34.71–34.75 GB/s** 범위로 안정적이었다. 이 수치를 RTX 3050의 보편적인 최대 성능으로 제시하지 않는다.

실험의 출력은 연속 64B metadata buffer다. Tensor Core의 특정 `mma.sp` register fragment에 직접 연결하는 shuffle/layout 비용이나 sparse MMA 자체는 측정하지 않았다. 소비자가 요구하는 per-lane metadata layout이 다르면 추가 redistribution이 필요할 수 있다. 입력 bit packing 형식을 자유롭게 바꿀 수 있는 경우의 모든 producer/consumer 공동 최적화도 이번 비교의 범위 밖이다.

**채택안:** 기존 연속 48B 형식과 독립 warp 변환에서는 `expansion<9>` / 128 threads를 기본으로 선택한다. L1 request 압박이 큰 fused kernel에서는 `expansion<7>`도 비교한다. Wider load라는 이유만으로 `uint4 + shuffle`을 채택하거나, SWAR decode 4명령만으로 전체 packed decoder 비용을 계산하지 않는다.

재현 명령과 결과 파일 안내는 [README.md](README.md), 구현은 [experiment.cu](experiment.cu), 집계는 [analyze.py](analyze.py)에 있다.
