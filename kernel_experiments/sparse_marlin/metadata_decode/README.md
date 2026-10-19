# Sparse-Marlin 3bit metadata: fetch vs matmul smoke test

2026-10-19, RTX 3050 Laptop 4GB (sm_86, 20 SM), CUDA 12.6, PyTorch 2.8.0+cu126. N=K=4096, group size=128, seed=20261019.

실제 Sparse-Marlin의 sparse INT4 가중치와 metadata를 사용했다. 원본 CUDA 소스를 별도 namespace로 생성하여 빌드하므로 `src/` 및 upstream 구현은 변경하지 않는다. `generated/provenance.json`에 원본 SHA256을 저장한다. `setup.py`가 커널 변경 사항의 재현 가능한 원본이다.

## 구현

- `baseline`: 기존 `marlin.mul_2_4`, 원래 4bit metadata.
- `fetch`: overlapping `ldmatrix.x2` 이후 `fetch_to_registers()`에서 정렬 + high/low 디코딩. 다음 타일의 decoded metadata를 double buffer에 둔다.
- `matmul`: register fetch에서는 compressed word 두 개만 유지하고, `matmul()` 시작에서 현재 타일의 정렬 + high/low 디코딩. j=0..3 및 batch-row loop에서 재사용한다.

두 수정본은 앞 실험의 `pair_decode<2>`를 사용한다. Low/high 함수를 각각 단일 PTX 블록으로 묶은 구현이며 매핑은 그대로다. 함수들이 inline되므로 최종 명령 순서는 컴파일러 스케줄링의 영향도 받는다. 소스 위치를 옮기는 실험이지 하드웨어에 절대적인 실행 위치를 강제하는 실험은 아니다.

`codec.cu`는 원래 물리적 metadata word layout을 offline GPU preprocessing으로 변환한다. 원래 warp별 64 words(256B)를 48 words(192B)로 압축한다. Lane p와 p+16의 각각 두 metadata word를 48bit로 인코딩한 뒤 두 lane의 96bit를 X/Y/Z planes에 저장한다. 겹치는 두 128B matrix를 읽는 `ldmatrix.x2` 결과는 lane당 X/Y 또는 Y/Z다. 이를 T0 `<<8`, T1 `>>8`로 정렬하고 high/low 디코딩한다.

첫 출력 word는 high 역매핑, 두 번째 word는 low 역매핑으로 인코딩한다. 두 매핑 모두 여섯 유효 2:4 패턴을 표현할 수 있으므로 원래 metadata를 손실 없이 저장할 수 있다. Unsupported nibble은 packing 단계에서 오류로 처리한다.

GMEM 및 shared stage 모두 metadata 크기가 25% 감소했다. 실제 metadata buffer는 **2,097,152 → 1,572,864 bytes**다. 전송은 16B cp.async를 유지하고 active copier를 128 → 96 threads로 줄였다. Shared stage stride, global stride, stripe pointer traversal도 압축 크기에 맞췄다. Dynamic shared allocation은 비교를 위해 원래 96KiB를 유지했다.

## 속도

CUDA Graph에 40 calls를 capture하고 3 replay 평균을 7회 무작위 순서로 측정한 중앙값, 단위 us/call. Weight packing/metadata 압축 및 정확도 검사는 timed region 밖이다. **이미 압축된 가중치의 GEMM 실행 시간**이며 매번 packing하는 one-shot 총시간은 아니다. GPU clock은 고정하지 않았다.

| Batch | Baseline | Fetch decode | Matmul decode | 정확도/안정성 |
|---|---:|---:|---:|---|
| 1 | 76.60 | 71.65 | 71.80 | 두 수정본 PASS, bit-exact |
| 8 | 75.00 | 69.89 | 69.48 | 두 수정본 PASS, bit-exact |
| 16 | 75.48 | 69.88 | 70.69 | 두 수정본 PASS, bit-exact |
| 32 | 80.71 | 75.84 | 76.39 | 수정본에 간헐적 오차 |
| 64 | 130.68 | 138.82 | 132.39 | 세 구현 모두 불안정 |
| 128 | 286.49 | 283.59 | 277.94 | 세 구현 모두 불안정 |

Batch 1~16에서 두 구현의 speedup은 약 1.067~1.080x다. Fetch와 matmul의 차이는 0.2~1.2% 수준이고 우열이 batch에 따라 바뀐다. 이 smoke test로 어느 위치가 일관되게 빠르다고 결론 내리지 않는다. 큰 batch의 시간은 오류가 발생한 구현의 관측값이므로, 정확한 연산의 속도 향상으로 해석하지 않는다.

## 정확도

원본 helper `gen_quant4_NT`가 만든 **동일 sparse/quantized weight**의 FP32 matmul을 oracle로 사용했다. TF32를 끄고 세 구현의 마지막 출력뿐 아니라 반복 간 변동 및 모든 반복의 최대 relative MAE를 확인했다. Relative MAE는 mean(abs(C-oracle))/mean(abs(oracle)). 판정은 finite, 최대 relative MAE < 0.002, 반복 결과 동일을 모두 요구한다.

실제 metadata 전체의 pack → unpack은 bit-exact였다. 다음 표는 profiler 없는 추가 **10회 반복**의 최대 relative MAE다. 단위는 %다.

| Batch | Baseline | Fetch decode | Matmul decode |
|---|---:|---:|---:|
| 1 | 0.01883 | 0.01883 | 0.01883 |
| 8 | 0.02010 | 0.02010 | 0.02010 |
| 16 | 0.02001 | 0.02001 | 0.02001 |
| 32 | 0.02256 | 0.25832 | 0.41921 |
| 64 | 1.06492 | 3.08212 | 3.50175 |
| 128 | 0.63910 | 1.75133 | 2.05928 |

Batch 1/8/16은 두 수정본 모두 10회 내내 원본과 bit-exact였다. Batch 32의 baseline은 이번 반복에서 안정적이었지만 두 수정본은 불안정했다. 따라서 이를 단순히 baseline의 오차와 동일하다고 취급하지 않는다. Batch 64/128은 baseline도 불안정하고 수정본의 오차가 더 크다.

Compute Sanitizer memcheck는 전체 batch에서 **0 errors**였다. 계측 실행에서는 모든 batch의 세 구현이 bit-exact였고 FP32 oracle 오차도 0.02% 부근이었다. 계측이 스케줄링을 바꿀 수 있으므로 이것을 일반 실행의 정확성 보장으로 사용하지 않는다. 원인은 이번 smoke 범위에서 특정하지 않았다. Packing의 손실과 일반 실행의 비결정적 출력은 구분해서 기록했다.

## 컴파일 자원

이번에 사용한 group_blocks=4의 ptxas 결과다. Spill 수치는 정적 ptxas 보고값으로 runtime instruction count가 아니다.

| Batch tile / N tile / K tile | Fetch regs | Matmul regs | Fetch spill store/load bytes | Matmul spill store/load bytes |
|---|---:|---:|---:|---:|
| 16 / 128 / 128 | 160 | 160 | 0 / 0 | 0 / 0 |
| 32 / 256 / 64 | 208 | 210 | 0 / 0 | 0 / 0 |
| 64 / 256 / 64 | 255 | 255 | 224 / 196 | 204 / 180 |

Batch 64/128에서 사용하는 tile은 register 한계 및 spill이 있다. 이것만으로 오차나 속도 차이의 원인을 단정하지 않는다. NCU 추가 profile은 이번 smoke 범위에서 실행하지 않았다.

## 재현

```bash
cd kernel_experiments/sparse_marlin/metadata_decode
make smoke
../../../kernel_bench/.venv/bin/python smoke.py --check-only --checks 10 --label repeat10
compute-sanitizer --tool memcheck --error-exitcode 1 \
  ../../../kernel_bench/.venv/bin/python smoke.py --check-only --label memcheck
```

원본 local extension이 없으면 상위 폴더에서 `make build`를 먼저 실행한다. 기본 smoke는 PASS/FAIL을 CSV에 기록하고 실패 case도 끝까지 측정한다. 실패 시 프로세스를 중단하는 correctness test가 아니다.

결과 파일:
- `results/summary.csv`, `results/timing.csv`: 중앙값과 126개 timing samples.
- `results/accuracy_check_repeat10.csv`: 10회 반복 정확도, worst MAE, 안정성.
- `results/accuracy.csv`: 최초 timing run의 3회 정확도. 최초 버전은 마지막 trial의 판정만 있으므로 전체 안정성 판단에는 repeat10 파일을 사용한다.
- `results/memcheck.log`, `results/resources.csv`, `results/build.log`: 검증 및 자원 사용량.
- `results/environment.json`: 실행 환경과 옵션.
