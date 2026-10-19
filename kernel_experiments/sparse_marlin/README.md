# Sparse-Marlin 수정 실험

제공된 `unpack24_to32`의 구현·의존성·실행 유닛 분석은 [`decoder_dependency/`](decoder_dependency/README.md)에 있습니다. 제공 매핑을 유지한 PTX 최적화 버전도 비교합니다.

원본 커널의 metadata 병목 분석은 [`kernel_profile/`](kernel_profile/README.md)에 있습니다. 실제 matmul의 NCU 지표와 metadata global/register 경로를 분리한 진단 실험을 보관합니다.

Metadata의 `ldmatrix.x2`와 인접 thread 쌍의 96-bit 압축 로딩 비교는 [`metadata_load/`](metadata_load/README.md)에 있습니다. 독립 microbenchmark와 실제 M=1 커널의 NCU 측정 결과를 보관합니다.

`src/`는 기존 `kernel_bench/Sparse-Marlin`의 소스 복사본입니다. 빌드 설정에는 공통 헤더의 변경 감지를 위한 `depends`를 추가했습니다. 이곳의 소스를 수정한 뒤 로컬 확장을 빌드하여 테스트합니다. 원본 출처는 [IST-DASLab/Sparse-Marlin](https://github.com/IST-DASLab/Sparse-Marlin), 기준 커밋은 `69c618a0a16084a0c12d91d0e5a42b69c55ccae2`이며 라이선스는 `src/LICENSE`에 보관합니다.

## 빠른 실행

저장소 루트에서:

```bash
cd kernel_experiments/sparse_marlin
make test                   # 빌드 + 소규모 6개 shape/group 조합 정확성 검사
make bench                  # 빌드 + 정확성 검사 + CUDA event 평균 시간
make bench ARGS="--m 16 --n 4096 --k 4096 --groupsize 128 --iters 200"
make upstream-test          # 원본 correctness suite (더 큰 allocation 포함)
```

기본 Python은 기존 `kernel_bench/.venv/bin/python`이며, 없으면 `python3`입니다. `make test PYTHON=/absolute/path/to/python`으로 바꿀 수 있습니다. 현재 환경의 PyTorch는 `2.8.0+cu126`, CUDA Toolkit은 12.6이며, 원본 빌드 설정의 타깃은 `sm_86`입니다. CUDA GPU가 노출된 환경에서 실행해야 합니다.

별도 가상환경이 필요하면 Python 3.12 및 CUDA Toolkit 12.6 환경에서:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install numpy setuptools wheel ninja
make test PYTHON="$PWD/.venv/bin/python"
```

## 수정할 곳

- `src/marlin/marlin_cuda_kernel_nm.cu`: 2:4 sparse matmul 커널, metadata load 및 파이프라인.
- `src/marlin/common/`: 공통 memory/MMA 코드.
- `src/marlin/marlin_cuda.cpp`: PyTorch 바인딩.
- `src/marlin/__init__.py`, `_semi_structured_conversions.py`: weight/scale/metadata packing.
- `src/setup.py`: 컴파일 옵션. 다른 GPU에서는 하드코딩된 `-arch=sm_86`을 직접 변경합니다.

수정 후 `make test` 또는 `make bench`를 다시 실행하면 `src/`에 확장을 재빌드합니다. 헤더 수정도 빌드 의존성에 반영되지만, 빌드 옵션이나 Python/PyTorch 버전 변경 후에는 `make clean`으로 초기화하세요. 빌드 병렬도는 `MAX_JOBS=2`가 기본입니다.

`run.py`는 로컬 Python 패키지와 `.so`의 경로를 확인하며, 실제 사용한 확장 경로를 stderr로 출력합니다. 정확성은 원본 quantization/packing helper로 생성한 2:4 INT4 weight의 FP16 matmul 기준값과 비교합니다(relative MAE < 0.002, 원본 테스트와 동일). 기본 검사는 batch 1/16/32와 groupsize -1/128을 다룹니다.

벤치마크의 stdout은 JSON Lines입니다. `mean_us`는 packing을 제외한 반복 matmul의 CUDA event 평균 시간이며 Python launch 간격이 포함될 수 있습니다. 작은 반복 실험용이며 전체 모델 성능을 나타내지는 않습니다.

```bash
mkdir -p results
make build
../../kernel_bench/.venv/bin/python run.py bench > results/baseline.jsonl
# 소스 수정 후 make build, 같은 명령으로 results/modified.jsonl 저장
```

실제 3bit metadata를 적용한 Sparse-Marlin의 fetch/matmul 디코딩 위치 비교는 [metadata_decode/README.md](metadata_decode/README.md)에 있다. K=N=4096 smoke에서 작은 batch는 원본과 bit-exact 및 약 1.07~1.08x였으며, batch 32 이상에서 관찰한 출력 불안정성도 함께 기록했다.
