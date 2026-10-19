# 원본 Sparse-Marlin의 metadata 병목 분석

Batch별 **전체 실행 시간 감소**를 비교한 smoke test는 [BATCH_BOTTLENECK.md](BATCH_BOTTLENECK.md)에 있다. 정확성 gate 없이 metadata GMEM 복사 제거와 LDSM 제거를 batch 1~256에서 비교한다.

K=4096에서 batch 1~256의 후속 smoke test는 [SMOKE_K4096.md](SMOKE_K4096.md)에 있다. 큰 batch 일부에서 정확성 기준 초과와 반복 실행 간 오차 변동을 관측했다.

2026-10-19, RTX 3050 4GB Laptop GPU (sm_86, 20 SM), PyTorch 2.8.0+cu126, CUDA 12.6, NCU 2024.3.2.

## 결론

**N=K=4096, M=1/16/32에서는 weight와 metadata의 global 전송이 주요 병목이다. Metadata의 shared → register `ldmatrix.x2` 처리량은 주요 병목이 아니다.**

독립 metadata 로딩 루프에서 관측한 LSU 99% 포화는 원본 matmul의 상태와 다르다. 원본은 LSU 약 14~15%, shared wavefront 처리율 약 11~14%, DRAM 처리율 약 84~87%였다. Metadata LDSM의 초과 shared wavefront는 0개다.

## 원본 측정

`../src/marlin/marlin_cuda_kernel_nm.cu`는 수정하지 않았다. 로컬 파일이 `kernel_bench/Sparse-Marlin` 원본과 byte 단위로 같고, 진단용 extension 생성 당시 SHA-256과도 일치하는지 확인했다.

입력은 원본 `gen_quant4_NT()`로 생성한 2:4 INT4 weight이며 group size=128이다. FP16 dense matmul 대비 relative MAE < 0.002를 확인한 후 NCU를 시작했다. 20회 warmup 뒤 `cudaProfilerStart/Stop`으로 커널 한 번만 선택했다.

| Shape (M,N,K) | NCU 시간, cache=all | DRAM %peak | LSU %peak | Shared wavefront %peak | Tensor cycle 활성도 |
|---|---:|---:|---:|---:|---:|
| 1,4096,4096 | 84.864 us | 85.89% | 14.53% | 12.20% | 14.91% |
| 16,4096,4096 | 86.688 us | 86.64% | 14.01% | 11.40% | 14.36% |
| 32,4096,4096 | 84.512 us | 84.90% | 14.09% | 13.89% | 27.34% |
| 1,1024,1024 | 12.768 us | 35.96% | 10.65% | 9.25% | 6.26% |

`cache-control=none`도 수집했다. 큰 shape의 원본 시간은 각각 85.152 / 86.112 / 85.696 us로 결론이 같았다. `none`은 캐시를 강제로 비우지 않는다는 뜻이며, 특정 working set의 완전한 L2 residency를 보장하지 않는다. Clocks는 잠그지 않았다. CUDA Graph timing과 NCU replay timing은 서로 다른 실험이므로 절대값을 섞어 비교하지 않는다.

원본 M=1 큰 shape는 162 registers/thread, 동적 shared 요청 96 KiB, achieved occupancy 약 18.29%였다. 이는 진단군과 비교할 때 기록할 자원 조건이며, 높은 DRAM 처리율과 전송 제거 실험을 함께 보아 병목을 판단했다.

## Metadata의 global → shared 경로

소스 325~331행의 `cp_async4_stream(...meta_ptr[i])`를 SASS `LDGSTS...128`과 연결했다. M≤16 기본 설정에서 metadata shared 영역의 stage 시작 offset은 `0x8400,0x8c00,0x9400,0x9c00`이다. M=32에서는 `0x8800,0x9000,0x9800,0xa000`이다.

Source CSV에서 PC를 중복 제거하고 predicate가 켜진 thread 수 × 16 B를 합산하면 N=K=4096의 metadata 복사량은 정확히 **2,097,152 B=2 MiB**다.

```text
압축 INT4 weight: K×N/4 bytes = 4 MiB
2:4 metadata:     K×N/8 bytes = 2 MiB
group scales:                    0.25 MiB
```

원본 M=1의 실제 DRAM read는 6,589,312 B, 약 6.28 MiB다. 진단군에서 metadata global 경로만 제거하면 DRAM read가 약 2 MiB 줄었다. Metadata는 weight+metadata 저장량의 1/3이고, 이 shape의 총 DRAM read에서 약 32%에 해당한다.

높은 DRAM %peak, 낮은 LSU %peak, metadata bytes 제거에 따른 실행 시간 감소를 함께 보면 이 경로에서는 global 명령 발행 한계보다 데이터 공급 비용이 중요하다.

## Metadata의 shared → register 경로

소스 384행의 `ldsm4_m()`은 `common/mem.h`의 `ldmatrix.m8n8.x2`이며 SASS에서는 `LDSM.16.M88.2`다. A의 로딩은 `.4`라 구분할 수 있다.

M=1,N=K=4096의 metadata register 로딩은:

- Dynamic `LDSM.16.M88.2`: 8,584 warp 명령.
- Shared wavefront: 17,168개, 명령당 정확히 2개.
- 초과 shared wavefront: **0개**.
- M=16도 동일하며 M=32에서도 초과 wavefront는 0개다.

레지스터 prefetch/drain 때문에 실제 MMA 작업 수보다 일부 load가 추가된다. 위 값은 NCU에서 관측한 실제 실행 횟수다.

Metadata load PC에 barrier stall이 보인다고 그 load가 bank conflict를 일으킨다고 해석하면 안 된다. M=1의 metadata load PC에서 barrier sample 155개가 관측되지만, 실제 excess wavefront는 0개다. 앞선 `wait_for_stage()`의 비동기 로딩 및 block 동기화 대기를 고려해야 한다.

원본 M=1의 long-scoreboard sample 627개 중 517개는 소스 353행 `__syncthreads()`에 연결된 BAR/WARPSYNC instruction에 잡혔다. 이 위치는 A/B/scale/metadata의 pipeline 준비를 함께 기다린다. Sample은 실행 시간 비율이나 특정 데이터만의 stall 비용이 아니므로, 이것만으로 metadata 단독 원인을 주장하지 않았다.

## 경로를 제거한 인과관계 실험

원본은 그대로 두고 별도 `metadata_diagnostics` extension에 세 가지 진단 커널을 생성했다. Metadata는 모든 위치에서 유효한 패턴 `0xE`(선택 위치 2,3)인 입력으로 통일했다. Packed INT4 weight는 무작위이고 scale은 0.01이다.

| 진단군 | 변경 |
|---|---|
| baseline | 원본 함수 그대로 실행 |
| no_gmem | metadata global 복사를 제거하고 shared metadata를 커널 시작 때 한 번 상수로 초기화. 기존 LDSM은 유지 |
| no_reg | global 복사는 유지하고 metadata LDSM만 상수 register로 대체 |
| no_all | metadata global 복사와 LDSM을 모두 제거 |

네 결과는 모든 측정 shape에서 baseline과 **bit-exact 일치**했다. 고정 metadata 조건에만 유효한 진단이며 일반 입력에 적용할 수 있는 구현 변경이 아니다. Compiler scheduling/register 할당도 달라질 수 있어 차이를 순수한 한 명령의 latency나 엄밀한 상한으로 해석하지 않는다.

Profiler 없이 CUDA Graph 안에 100회 호출을 넣고, 측정당 5회 replay했다. 4개 variant × 9 samples를 무작위 순서로 실행했다. 표는 sample 중앙값이다.

| Shape | baseline | no_gmem | no_reg | no_all |
|---|---:|---:|---:|---:|
| M=1,N=K=4096 | 74.16 us | **52.39 us** | 74.27 us | 52.06 us |
| M=16,N=K=4096 | 75.15 us | **53.35 us** | 75.26 us | 52.84 us |
| M=32,N=K=4096 | 81.68 us | **63.23 us** | 81.65 us | 61.10 us |
| M=1,N=K=1024 | 10.39 us | 9.50 us | 9.92 us | 8.97 us |

큰 shape의 M=1/16에서는 global metadata를 제거하면 시간이 약 29% 줄고, LDSM만 제거하면 거의 변하지 않는다.

NCU에서도 경로 제거를 확인했다. M=1의 uniform 입력, cache=all 기준:

| Variant | NCU時間 | DRAM read | 全LDSM命令 | Registers/thread |
|---|---:|---:|---:|---:|
| baseline | 83.968 us | 6,588,800 B | 25,752 | 162 |
| no_gmem | 57.088 us | 4,491,776 B | 25,752 | 160 |
| no_reg | 84.000 us | 6,588,544 B | 17,168 | 158 |
| no_all | 59.328 us | 4,491,008 B | 17,168 | 135 |

`no_reg`에서는 metadata LDSM 8,584개만 사라지고, metadata global copy의 issued bytes는 2 MiB로 유지된다. `no_gmem`에서는 metadata global copy가 사라지고 metadata LDSM은 남는다. 원래의 shared 할당량은 변경하지 않았다.

작은 shape에서는 global 복사 제거의 개선이 8.6%, register 로딩 제거는 4.6%였다. DRAM 이용률도 낮으므로 이 shape까지 같은 DRAM 대역폭 병목이라고 일반화하지 않는다. 고정비, pipeline fill/drain, 동기화 중 무엇이 지배적인지는 추가 분리가 필요하다.

## 압축 구현에 대한 시사점

이 shape에서 3-bit metadata를 사용하면 global 전송이 0.5 MiB 줄어든다. 원래 총 DRAM read의 약 8%로, byte 양만 보면 약 8%의 시간 단축, 약 1.09×가 참고값이다. Decode 비용을 무시한 대역폭 모델이며 실제 측정치나 보장값은 아니다.

우선순위는 metadata global 전송을 줄이면서 기존 LDSM의 한 명령·무충돌 특성을 유지하고, 실제 pipeline에서 추가 decode 비용을 측정하는 것이다. 독립 microbenchmark의 decode 배수를 원본 matmul 전체에 직접 적용하면 안 된다. 원본의 ALU 처리량에는 여유가 있지만 의존성과 register 사용량에 따라 성능이 달라질 수 있다.

## 검증 범위와 재현

- 원본 정확성: M=1/16/32,N=K=4096 및 M=1,N=K=1024 통과.
- 진단군: 위 4개 shape에서 baseline과 bit-exact 일치.
- 4 shapes × 4 variants × 9 samples = 144개 CUDA Graph timing sample.
- 8개 packed baseline NCU report + 8개 uniform diagnostic report.
- Shared 초기화를 추가한 no_gmem 경로의 memcheck와 synccheck: 모두 0 errors. 검사 전 모든 진단 variant의 출력도 비교한다.
- M=64의 packed 입력은 최초 FP16 기준 relative MAE 0.00579로 기준 0.002를 초과해 제외했다. 별도 seed의 FP32 기준 재확인에서도 약 0.0112가 관측됐다. 원인 분석·수정은 이번 metadata 성능 분석 범위에 포함하지 않았다.

```bash
cd kernel_experiments/sparse_marlin/kernel_profile
make build
make profile
make bench
make check
make analyze
```

기본 Python은 기존 `kernel_bench/.venv`다. GPU가 노출된 환경에서 실행해야 한다. `profile.py`는 이미 raw/source CSV가 있는 case를 건너뛰므로 다시 측정할 때 해당 결과를 다른 폴더로 보관하거나 삭제해야 한다.

Source CSV는 같은 SASS PC가 여러 inline source 위치에 반복될 수 있어 PC별로 중복 제거했다. `analyze.py`는 시간 단위를 us, DRAM read를 byte로 정규화한다.

주요 산출물은 `results/counter_summary.csv`, `metadata_source_summary.csv`, `timing_summary.csv`다. 원본 SASS/소스별 counter는 각 `*_source.csv`, 모든 metric과 단위는 `*_raw.csv`에 있다. 생성한 진단 CUDA는 `generated/`에 있으며 `provenance.json`에 원본 경로와 SHA-256을 기록한다.
