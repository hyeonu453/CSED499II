# Sparse-Marlin 3bit metadata: fetch vs matmul smoke test

## TL;DR

동일 SM 환경에서 CSED499II 저장소 루트 기준으로 실행한다. 시스템에 CUDA Toolkit 12.6 (`nvcc`)과 C++ 컴파일러가 설치되어 있어야 한다.

```bash
cd /path/to/CSED499II

uv sync --project kernel_bench --locked
export BENCH_PYTHON="$PWD/kernel_bench/.venv/bin/python"
uv pip install --python "$BENCH_PYTHON" setuptools wheel ninja

make -C kernel_experiments/sparse_marlin \
  PYTHON="$BENCH_PYTHON" build
make -C kernel_experiments/sparse_marlin/metadata_decode \
  PYTHON="$BENCH_PYTHON" build

"$BENCH_PYTHON" kernel_experiments/sparse_marlin/metadata_decode/smoke.py \
  --checks 10
```

결과는 `kernel_experiments/sparse_marlin/metadata_decode/results/`의 `summary.csv`(속도)와 `accuracy.csv`(정확도, `passed` 열)에 저장된다.

2026-10-19, RTX 3050 Laptop 4GB (sm_86, 20 SM), CUDA 12.6, PyTorch 2.8.0+cu126. N=K=4096, group size=128, seed=20261019.

실제 Sparse-Marlin의 sparse INT4 가중치와 metadata를 사용했다. 원본 CUDA 소스를 별도 namespace로 생성하여 빌드하므로 `src/` 및 upstream 구현은 변경하지 않는다. `generated/provenance.json`에 원본 SHA256을 저장한다. `setup.py`가 커널 변경 사항의 재현 가능한 원본이다.

## CSED499II 전체 저장소 + uv로 실행

전체 `CSED499II` 저장소를 가져오고 **기존 `kernel_bench`의 uv 환경 하나를 공유**하는 절차다. 새 venv를 따로 만들거나 torch를 pip로 별도 설치할 필요가 없다. 동일한 `sm_86` GPU라면 아키텍처 설정도 그대로 사용한다.

### 1. 소스와 시스템 환경

저장소 루트 `CSED499II/`로 이동한다. 아래 파일/폴더가 포함된 commit을 가져와야 한다. 로컬에서만 작성하고 commit/push하지 않은 실험 파일은 git clone에 포함되지 않는다.

```text
CSED499II/
├── kernel_bench/
│   ├── pyproject.toml
│   ├── uv.lock
│   └── .python-version
└── kernel_experiments/sparse_marlin/
    ├── Makefile
    ├── src/                        # baseline kernel과 weight packer
    ├── decoder_dependency/
    │   ├── decoder.cuh
    │   └── pair.cuh
    └── metadata_decode/            # setup.py, codec.cu, smoke.py, Makefile
```

Linux, `uv`, NVIDIA driver, CUDA Toolkit 12.6의 `nvcc`, C++ compiler와 `make`가 필요하다. 검증 환경은 Python 3.12 / PyTorch 2.8.0+cu126 / RTX 3050 Laptop (sm_86, 4GB)이다. uv는 Python 패키지를 설치하지만 시스템 CUDA Toolkit과 driver는 설치하지 않는다. 현재 커널은 block당 opt-in shared memory 96KiB를 사용한다.

Toolkit이 `/usr/local/cuda-12.6`에 설치되어 있다면:

```bash
cd /path/to/CSED499II
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
nvcc --version
```

`CUDA_HOME`은 실제 설치 경로에 맞춘다. `CUDA_VISIBLE_DEVICES`로 고른 GPU는 스크립트에서 device 0으로 보인다.

### 2. kernel_bench uv 환경 복원

이후 명령은 모두 **CSED499II 루트**에서 실행한다.

```bash
uv sync --project kernel_bench --locked
export BENCH_PYTHON="$PWD/kernel_bench/.venv/bin/python"
uv pip install --python "$BENCH_PYTHON" setuptools wheel ninja
```

`kernel_bench/pyproject.toml`은 Python `>=3.12,<3.13`, torch `2.8.0` 및 CUDA 12.6용 PyTorch index를 지정한다. `uv sync --locked`는 저장소의 `uv.lock`을 유지하며 `kernel_bench/.venv`를 복원한다. 마지막 명령은 CUDA extension 빌드 도구를 그 환경에 추가한다.

이 벤치마크에는 `kernel_bench/build.sh`를 실행할 필요가 없다. 해당 스크립트는 다른 커널들까지 설치한다. 여기서는 아래 두 local extension만 빌드한다. 나중에 다시 `uv sync`하여 추가 빌드 도구가 제거됐다면 `uv pip install` 명령을 다시 실행한다. 추가 패키지를 유지하려면 `uv sync --project kernel_bench --locked --inexact`를 사용할 수 있다.

환경 확인:

```bash
"$BENCH_PYTHON" -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

### 3. baseline과 수정본 빌드

```bash
make -C kernel_experiments/sparse_marlin PYTHON="$BENCH_PYTHON" build
make -C kernel_experiments/sparse_marlin/metadata_decode PYTHON="$BENCH_PYTHON" build
```

첫 번째는 원본 Sparse-Marlin, 두 번째는 fetch/matmul decoder와 codec을 빌드한다. 별도 `pip install marlin`은 필요하지 않다. `smoke.py`가 형제 `src/`를 import한다. 다른 머신의 `.so`와 `build/`는 재사용하지 않고 대상 환경에서 다시 빌드한다. 빌드 자원이 부족하면 각 make에 `MAX_JOBS=1`을 추가한다.

현재 두 setup.py는 `-arch=sm_86`을 명시한다. 동일 SM이면 수정하지 않는다. 다른 아키텍처로 이식하려면 `src/setup.py`와 `metadata_decode/setup.py`의 해당 플래그를 모두 바꾸고, 기존 build 및 extension을 제거한 뒤 검증해야 한다. `TORCH_CUDA_ARCH_LIST`만 설정해서는 명시된 `-arch`가 바뀌지 않는다. 실제 검증한 아키텍처는 sm_86이다.

### 4. 실행

```bash
"$BENCH_PYTHON" kernel_experiments/sparse_marlin/metadata_decode/smoke.py --checks 10
```

이 명령 하나로 baseline / fetch decode / matmul decode를 모두 비교한다. 기본 batch는 `1 8 16 32 64 128`, timing은 7 samples, graph당 40 calls, sample당 3 replay다. N=K=4096과 group size=128은 현재 `smoke.py`에 고정돼 있다. Packing 비용은 timing 밖이다.

작은 batch부터 확인하려면:

```bash
"$BENCH_PYTHON" kernel_experiments/sparse_marlin/metadata_decode/smoke.py \
  --batches 1 8 16 --checks 10
```

정확도만 확인하거나 timing 반복을 바꾸려면:

```bash
"$BENCH_PYTHON" kernel_experiments/sparse_marlin/metadata_decode/smoke.py \
  --check-only --checks 10 --label repeat10

"$BENCH_PYTHON" kernel_experiments/sparse_marlin/metadata_decode/smoke.py \
  --samples 9 --calls 100 --replays 5
```

직접 `.venv/bin/python`을 호출하므로 실행 시 uv의 추가 sync가 발생하지 않는다. 가상환경 activation도 필요하지 않다. 새 shell에서는 CSED499II 루트에서 `BENCH_PYTHON`을 다시 지정하면 된다.

선택 사항인 메모리 검사는 일반 timing과 별도로 실행한다. 일반 실행에 NCU나 Compute Sanitizer는 필요하지 않다.

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
  "$BENCH_PYTHON" kernel_experiments/sparse_marlin/metadata_decode/smoke.py \
  --check-only --label memcheck
```

### 5. 결과 확인

현재 shell 위치와 무관하게 `kernel_experiments/sparse_marlin/metadata_decode/results/`에 저장된다.

| 파일 | 내용 |
|---|---|
| `summary.csv` | batch/구현별 `median_us`, `speedup_vs_baseline` (1보다 크면 빠름) |
| `timing.csv` | 각 sample의 μs/call. 기본 설정에서 126행 |
| `accuracy.csv` | `worst_relative_mae`, 반복 안정성 `stable`, 최종 판정 `passed` |
| `environment.json` | GPU, PyTorch/CUDA 버전, 옵션, metadata 크기 |
| `accuracy_check_repeat10.csv` | `--check-only --label repeat10` 결과 |

`passed=True`는 모든 반복이 finite이고 최대 relative MAE < 0.002(0.2%)이며 반복 출력이 같다는 뜻이다. `vs_baseline_equal_pct`는 마지막 출력끼리의 비교이므로 이것만으로 반복 전체의 정확도를 판정하지 않는다. Relative MAE `0.0002`는 0.02%다.

정확도 실패가 있어도 나머지 batch를 측정하고 정상 종료할 수 있으므로 **exit code 0만으로 PASS라고 판단하지 말고 CSV의 `passed`를 확인한다.** Packing roundtrip 불일치나 CUDA 오류는 예외로 종료한다.

재실행은 같은 이름의 파일을 덮어쓴다. `--label`은 accuracy/timing/environment 파일에 suffix를 붙이지만 `summary.csv`에는 붙이지 않으므로 비교할 실행은 결과 폴더를 별도로 보관한다. GPU clock은 고정하지 않으니 다른 GPU 작업을 종료하고 GPU 모델/driver/clock 및 power 상태도 함께 기록한다.

현재 관찰한 제한: batch 1/8/16은 원본과 bit-exact였으나 수정본의 batch 32 및 세 구현의 batch 64/128에서 비결정적 오차가 발생했다. 대상 머신에서도 이 구간의 시간과 정확도를 함께 확인한다.

### 문제 해결

- `No module named marlin_cuda`: 같은 `BENCH_PYTHON`으로 baseline make를 실행한다.
- `No module named metadata_decode`: 수정본 make를 실행한다.
- `No module named setuptools` / ninja 경고: 위 `uv pip install --python ... setuptools wheel ninja`를 실행한다.
- `nvcc` 없음 / CUDA mismatch: `CUDA_HOME`, `nvcc --version`, `torch.version.cuda`를 확인한다. 재현 조합은 CUDA 12.6이다.
- `no kernel image` / `invalid device function`: 두 setup.py의 `-arch`와 실제 GPU capability를 확인한다.
- `undefined symbol` / extension import 오류: 다른 환경의 `.so`를 재사용하지 말고 두 extension을 현재 환경에서 재빌드한다.
- Header 수정이 반영되지 않음: 해당 extension의 `build/`와 `.so`를 제거하고 재빌드한다. distutils의 header 변경 감지가 누락될 수 있다.


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

## 기존 측정 아티팩트

결과 파일:
- `results/summary.csv`, `results/timing.csv`: 중앙값과 126개 timing samples.
- `results/accuracy_check_repeat10.csv`: 10회 반복 정확도, worst MAE, 안정성.
- `results/accuracy.csv`: 최초 timing run의 3회 정확도. 최초 버전은 마지막 trial의 판정만 있으므로 전체 안정성 판단에는 repeat10 파일을 사용한다.
- `results/memcheck.log`, `results/resources.csv`, `results/build.log`: 검증 및 자원 사용량.
- `results/environment.json`: 실행 환경과 옵션.
