# K=4096 batch smoke test

2026-10-19, RTX 3050 4GB Laptop GPU, 원본 Sparse-Marlin, N=K=4096, group size=128, seed=20261019.

원본 helper로 weight를 한 번 생성하고 모든 batch에서 공유했다. 입력도 batch 256개를 한 번 생성하고 prefix를 사용했다. FP32 matmul(TF32 비활성)을 기준으로 같은 입력을 3회 실행했다. Relative MAE < 0.002(0.2%)이고 출력이 finite이면 통과다. 이 표는 단일 seed의 smoke test이며 완전한 정확성 보장이 아니다.

| Batch | Relative MAE 범위 (%) | 결과 | 시간 (us) |
|---|---:|---|---:|
| 1 | 0.0188 | PASS | 78.08 |
| 2 | 0.0194 | PASS | 78.59 |
| 4 | 0.0196 | PASS | 78.48 |
| 8 | 0.0198 | PASS | 78.90 |
| 16 | 0.0198 | PASS | 77.00 |
| 24 | 0.0222 | PASS | 82.48 |
| 32 | 0.0224 | PASS | 84.93 |
| 48 | 0.1734–0.2940 | FAIL | — |
| 64 | 0.7241–1.0034 | FAIL | — |
| 96 | 0.3067–0.8392 | FAIL | — |
| 128 | 0.4258–0.5759 | FAIL | — |
| 192 | 0.0358–0.2153 | FAIL | — |
| 256 | 0.0461–0.0816 | PASS | 699.08 |

8/13 통과. 실패한 batch에서는 동일한 입력·weight를 반복 실행해도 오차가 변했다. Batch 256도 오차가 변하지만 이번 3회에서는 기준 이내였다. 특정 batch 경계 이상이 모두 실패한다고 일반화할 수 없으며, 큰 batch 경로의 정확성·동기화를 조사할 증상이다. 원인은 아직 특정하지 않았다.

시간은 profiler 없이 20회 호출을 CUDA Graph로 capture한 뒤, replay를 3회 측정한 per-call 중앙값이다. 실패 케이스도 진단 목적으로 측정해 raw CSV에는 남겼지만 유효한 성능 수치로 사용하지 않는다. Clocks는 잠그지 않았다.

```bash
cd kernel_experiments/sparse_marlin/kernel_profile
make smoke
```

스크립트는 `smoke.py`, raw 결과와 환경은 `results/smoke_k4096/results.csv`, `environment.json`에 있다. 실패 결과도 모두 저장하기 위해 스크립트는 결과 행의 `passed` 필드로 통과 여부를 기록한다.
