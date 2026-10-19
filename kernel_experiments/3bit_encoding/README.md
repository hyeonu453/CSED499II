# sm_86 metadata expansion experiments

실험 결과와 해석은 [REPORT.md](REPORT.md), CUDA 구현은 [experiment.cu](experiment.cu)에 있습니다.

```bash
cd kernel_experiments/3bit_encoding
make all
make run       # correctness + randomized CUDA Graph throughput + latency
make profile   # run separately: profiling interferes with timing
make check     # compute-sanitizer memcheck and synccheck
make analyze
```

CUDA Toolkit 12.6, C++17, sm_86으로 실행했습니다. 외부 Python 패키지는 필요 없습니다.
`make run`은 48 MiB 입력과 64 MiB 출력을 사용하며 GPU 점유 시간이 수 분 걸릴 수 있습니다.
`./experiment verify`는 짧은 correctness 전용 실행입니다.
`./experiment profile 9`는 direct_u32_4 profiling 대상 kernel을 실행합니다.

- `results/measurements.csv`: CUDA Graph 기반 최종 raw measurements.
- `results/measurements_host_launch.csv`: 초기 host launch 기반 비교 기록. 최종 표에는 사용하지 않음.
- `results/summary.csv`: 5회 throughput / 9회 warp latency / 7회 decoder 측정 중앙값과 범위.
- `results/ncu_0.csv` ... `ncu_9.csv`: 실제 hardware counter 원본.
- `results/profile_summary.csv`: counter 원본과 warp당 정규화 값.
- `results/sass.txt`, `*.sass`, `sass_inventory.json`: 실제 SASS 및 데이터 의존 경로.
- `results/build.log`, `run.log`, `memcheck.log`, `synccheck.log`: 컴파일 및 correctness 기록.
- `results/telemetry.csv`: 측정 후반부터 기록한 GPU 클록/전력/온도. 전체 실행과 1:1 대응되는 trace는 아님.

입력의 code i는 bit offset 3*i, 출력은 4*i에 저장합니다(LSB-first, little endian).
한 warp당 입력 stride=48 bytes, 출력 stride=64 bytes입니다.
method 5만 16개 24-bit 그룹의 byte plane을 transpose한 별도 저장 형식을 사용합니다.
