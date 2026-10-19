# Batch별 전체 실행 시간: metadata GMEM 복사와 LDSM 제거 비교

2026-10-19, RTX 3050 4GB Laptop GPU. N=K=4096, group size=128. 정확성 gate 없이 batch별 전체 matmul 실행 시간 감소를 비교한 진단 smoke test다.

## 방법

기존 진단군을 사용한다. Metadata는 0xE로 고정하고 packed weight는 무작위다.

- baseline: 원본 Sparse-Marlin.
- no_gmem: metadata GMEM 복사를 제거하고 shared에 상수 metadata를 한 번 초기화한다. Metadata LDSM은 유지한다.
- no_reg: metadata GMEM 복사는 유지하고 metadata LDSM을 상수 register로 대체한다.
- no_all: 두 metadata 경로를 모두 제거한다.

13개 batch에서 각 variant를 CUDA Graph에 20회 capture하고, sample마다 3회 replay했다. Variant별 5개 sample을 무작위 순서로 실행했다. 표는 전체 matmul의 per-call 시간 중앙값이다. Profiler를 붙이지 않은 timing이며, clocks는 잠그지 않았다. 13×4×5=260개 timing sample이다.

사용자 요청대로 정확성 결과로 중단하지 않는다. 고정 metadata 진단군이며 production 구현이나 일반 metadata 입력에 대한 정확성 결과로 해석하지 않는다. Compiler scheduling/register 할당 변화도 포함하므로 순수한 단일 명령 latency를 측정한 결과는 아니다.

## 전체 실행 시간

양수 감소율은 baseline보다 빨라졌다는 뜻이다. 1% 내외의 작은 차이는 이 smoke test에서 뚜렷한 개선이라고 해석하지 않는다.

| Batch | Baseline us | no_gmem us | no_reg us | no_all us | GMEM 제거 시간 감소 | LDSM 제거 시간 감소 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 77.81 | 55.88 | 77.79 | 55.62 | 28.19% | 0.02% |
| 2 | 77.94 | 55.40 | 77.29 | 55.48 | 28.92% | 0.83% |
| 4 | 78.03 | 56.06 | 78.11 | 55.77 | 28.15% | -0.11% |
| 8 | 77.94 | 56.12 | 78.05 | 55.94 | 28.01% | -0.13% |
| 16 | 77.21 | 56.41 | 78.11 | 56.05 | 26.95% | -1.17% |
| 24 | 79.94 | 65.89 | 80.47 | 65.26 | 17.57% | -0.66% |
| 32 | 82.21 | 66.61 | 82.24 | 65.89 | 18.97% | -0.04% |
| 48 | 101.44 | 98.68 | 99.36 | 95.56 | 2.73% | 2.05% |
| 64 | 139.21 | 134.98 | 138.12 | 129.18 | 3.04% | 0.78% |
| 96 | 229.24 | 204.85 | 220.19 | 198.49 | 10.64% | 3.95% |
| 128 | 287.80 | 259.04 | 286.28 | 259.12 | 9.99% | 0.53% |
| 192 | 481.81 | 415.35 | 478.12 | 410.59 | 13.79% | 0.77% |
| 256 | 701.92 | 615.94 | 707.12 | 612.98 | 12.25% | -0.74% |

## 대표 batch의 NCU

원본 baseline을 uniform 입력으로 20회 warmup하고 profiler start/stop으로 한 번만 측정했다. Cache control은 none이며 timing 비교와는 별도의 실행이다.

| Batch | DRAM %peak | LSU %peak | Tensor cycle 활성도 | Registers/thread | Local load 명령 | Local store 명령 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 87.54% | 13.59% | 13.95% | 162 | 0 | 0 |
| 32 | 84.94% | 13.25% | 25.71% | 208 | 0 | 0 |
| 48 | 74.07% | 13.88% | 32.55% | 255 | 496 | 480 |
| 64 | 70.71% | 13.17% | 33.51% | 255 | 11,368 | 4,128 |
| 256 | 93.47% | 9.43% | 25.91% | 255 | 43,852 | 5,776 |

Local 명령은 warp 단위 dynamic instruction count이며 byte 양이 아니다. Register 상한과 spill의 존재는 다른 성능 요인이지만, 이 관측만으로 큰 batch의 모든 비용을 spill에 귀속하지 않는다.

## 해석

- Batch 1~16: metadata global 전송 제거로 약 27~29% 단축된다. LDSM만 제거하면 개선은 거의 없다. Metadata 최적화 대상은 global 쪽이다.
- Batch 24/32: global 쪽 제거 효과는 약 18~19%다. LDSM 제거 효과는 거의 없다.
- Batch 48/64: metadata의 어느 한 경로를 제거해도 단축은 약 1~3%다. Metadata 단독을 전체의 주요 제한이라 할 수 없으며 둘 다 제거해도 약 6~7%다. Register 사용과 spill을 포함한 다른 요인을 더 분리해야 한다.
- Batch 96~256: metadata global 전송 제거 효과는 약 10~14%, LDSM만의 효과는 약 -1~4%다. Batch 256에서는 DRAM %peak가 93.5%이므로 global 쪽 압력이 강하지만 그 전부가 metadata 때문은 아니다.

**이번에 테스트한 batch에서 metadata ldmatrix 제거가 큰 개선을 주는 구간은 발견하지 못했다. 작은 batch일수록 metadata global 전송의 영향이 크다. 큰 batch를 일괄적으로 같은 metadata GMEM 병목으로 일반화할 수도 없다.**

## 재현과 데이터

```bash
cd kernel_experiments/sparse_marlin/kernel_profile
make batch-bottleneck
```

Timing만 측정하려면 `batch_bottleneck.py`, 소수 metric의 NCU만 수집하려면 `batch_counters.py`를 실행한다. 출력은 `results/batch_bottleneck_k4096/`의 `measurements.csv`, `summary.csv`, `ncu_summary.csv`다. 각 NCU raw CSV에는 metric의 unit 행도 저장한다.
