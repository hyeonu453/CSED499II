# 제공된 unpack24_to32의 dependency와 실행 유닛 분석

2026-10-19, RTX 3050 4GB Laptop GPU, sm_86, 20 SM, CUDA 12.6, NCU 2024.3.2.

## 구현과 매핑

`decoder.cuh`의 `unpack24_to32()`는 제공 코드의 연산과 LUT 상수를 그대로 유지한다. 기존 실험의 매핑을 정답으로 강제하지 않았다. 이 코드 자체가 정의하는 nibble 매핑은 다음과 같다.

```text
code:   0  1  2  3  4  5  6  7
output: C  D  E  A  8  9  4  0
```

8개의 3-bit code를 하위 24-bit에 LSB-first로 넣으면, 각 code의 결과가 대응하는 nibble로 나온다. 이 매핑을 lookup하는 독립 scalar reference와 비교했다.

GPU에서 **2^24개 전체 packed 입력**에 대해 원본 함수, 입력 상위 8-bit에 garbage를 추가한 경우, spreading만, nibble decode만, 두 최적화 함수 모두 mismatch 0이었다. 이는 제공 매핑의 구현과 각 nibble의 독립성에 대한 검증이다. 매핑 자체를 특정 sparse metadata 규약에 고정하지 않는다.

## 실제 dependency

한 입력 word 안에서의 논리적 의존은 다음과 같다.

```text
x → shift4 → LOP3 → shift2 → LOP3 → shift1 → LOP3 → mask
                                                        ├→ SHR1 → low LOP3 ──┐
                                                        └→ SHL1,SHL2 → high LOP3 ─┤
                                                                                 combine
```

세 spreading 단계는 앞 단계의 결과를 다음 단계가 사용하므로 직렬이다. Low/high decode는 spreading 후 서로 독립적으로 계산할 수 있지만 마지막 combine은 둘 다 기다린다.

실제 probe SASS에서 global load/store와 주소 계산을 제외하고 input register→output register DAG를 추적했다. 원본·두 최적화 버전 모두 최장 의존 깊이는 **10개 명령**이다. 이는 명령 개수 기준 depth이며 cycle 예측값이 아니다. 명령별 latency와 같은 실행 유닛의 발행 경쟁이 별도로 영향을 준다. 상세 DAG와 실제 critical path는 `results/dependency_dags.json`에 있다.

원본 probe의 input-dependent 명령은 15개다.

```text
5 × IMAD.SHL
1 × SHF
9 × LOP3
```

원본 코드에 쓰인 shift가 항상 SHF로 나오는 것은 아니다. 또한 `(x & 0x77777777) << n`이 `(x << n) & shifted_mask` 형태로 바뀌면서 high decode 앞에 mask LOP3 두 개가 추가됐다. 최종 SASS에는 `0xEEEEEEEE`, `0xDDDDDDDC` mask가 보인다. LUT 0xD8이 SASS에서 operand 순서 변경과 함께 0xB8로 나오는 것은 같은 연산이다.

## 같은 매핑을 유지한 비교 구현

- `provided_full`: 원본의 여러 C++/inline PTX 블록.
- `fused_ptx`: 모든 단계를 한 PTX 블록에 넣고 mask 이후 shift를 유지한다. 추가 mask 두 개를 제거해 13개 명령이 된다.
- `forced_shf`: 위 버전에서 shift를 `shf.*`로 명시한다. 역시 13개 명령이지만 실행 유닛 배분이 달라진다.
- `spreading_only`, `nibble_decode_only`, `rotate_control`: 구성 요소와 loop overhead를 보기 위한 대조군. 전체 decoder와 같은 의미의 함수가 아니다.

`unpack24_fused<false>()`와 `<true>()`가 각각 fused 및 forced-SHF 구현이다. 매핑은 두 버전 모두 원본과 같다.

Probe와 반복 kernel은 compiler context가 달라 shift의 IMAD/SHF 선택이 다를 수 있다. NCU로 측정한 ILP=8 반복 kernel의 decoder당 steady-state 명령은:

| 구현 | LOP3 | SHF | IMAD.SHL | 합계 |
|---|---:|---:|---:|---:|
| 원본 | 9 | 1 | 5 | 15 |
| Fused PTX | 7 | 1 | 5 | 13 |
| Forced SHF | 7 | 6 | 0 | 13 |

Dynamic instruction 총수에서도 fused는 원본보다 decoder당 약 2개 적다. Setup/teardown의 몇 개 명령 차이는 별도다.

## 단일 warp: dependency와 ILP

계산은 register에서만 반복한다. 256 loops × 16 unroll이며 출력은 다음 입력으로 피드백된다. 외부 load/store는 timed loop 밖이다. 각 lane에 1/2/4/8개의 독립 register chain을 둔다. 7개 sample을 무작위 순서로 실행했다.

표의 단위는 **warp-wide decoder 호출 하나당 cycles**다. 각 호출은 lane당 code 8개를 처리한다. ILP=2 열은 두 chain 전체 시간을 decoder 호출 수 2로 나눈 값이다. Timer/loop 비용을 포함하며 순수 함수 latency와 완전히 같지는 않다. Rotate 대조군은 단일 chain에서 4.505 cycles/call이었다.

| 구현 | ILP 1 | ILP 2 | ILP 4 | ILP 8 |
|---|---:|---:|---:|---:|
| 원본 | 53.380 | 30.221 | 22.002 | 20.141 |
| Fused PTX | 50.443 | 25.815 | 17.814 | 16.212 |
| Forced SHF | 45.443 | 26.690 | 26.236 | 26.165 |

원본의 chain 하나는 dependency에 민감하다. ILP를 늘리면 독립 연산이 겹치지만 ILP 4~8에서는 개선이 줄어 실행 유닛 처리량 한계가 드러난다. Metadata 64-bit를 만들기 위해 두 24-bit 입력을 복원하는 경우에는 ILP=2 결과가 직접적인 참고 조건이다. 이것도 full matmul의 추가 latency 측정치는 아니다.

## 여러 warp: ALU 처리량 제한

GPU 포화 실험은 160 blocks × 128 threads에서 수행했다. 아래는 ILP=8의 NCU 결과다. 모든 구현의 논리 호출 수는 20,971,520 warp-decoder calls로 같다.

| 구현 | ALU cycle 활성도 | FMA Heavy cycle 활성도 | LSU 처리율 | NCU SM cycles | 원본 대비 비용 |
|---|---:|---:|---:|---:|---:|
| 원본 | 99.63% | 49.74% | 0.003% | 5,271,018.5 | 1.000 |
| Fused PTX | 99.54% | 62.09% | 0.003% | 4,222,156.5 | 0.801 |
| Forced SHF | 99.87% | 0.003% | 0.002% | 6,832,924 | 1.296 |

Percent는 각 pipeline metric의 기준에 대한 값이며 서로 더해서 100%가 되는 비율이 아니다. FMA Heavy 이용률이 떠도 이 decoder가 부동소수점 연산을 하는 것은 아니다. `IMAD.SHL`이 해당 유닛을 사용한다. [NVIDIA pipeline 설명](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#pipelines)

**충분한 ILP에서는 ALU가 주요 처리량 제한이다.** Fused는 ALU에서 실행되는 두 mask 명령을 줄여 약 20%의 cycle 비용을 절감했다. 반면 SHF 강제는 단일 chain latency를 줄이지만 shift까지 ALU에 몰려 처리량이 원본보다 약 30% 나빠졌다.

원본 ILP=8의 sampling은 math-pipeline throttle 34,017/102,877 samples, wait 8,673 samples였다. ILP=1에서는 wait가 5,498/14,775 samples로 상대적으로 더 컸다. Stall sample을 cycle 비율이나 특정 명령의 latency로 직접 해석하지 않고 ILP·SASS·pipeline 이용률과 함께 판단했다.

ILP=2에서도 fused의 전체 SM cycles는 원본 1,376,251→1,119,472로 약 18.7% 줄었다. Raw와 fused의 register 수는 single-warp ILP=2에서 모두 16/thread, ILP=8에서 모두 30/thread다. 주요 NCU 비교군의 local load/store는 모두 0으로 spill은 없었다.

## 실제 matmul에 넣을 때 확인할 지점

1. 두 metadata word의 decoder를 독립적으로 배치해 word 간 ILP를 확보한다.
2. Decode→MMA의 직접 의존 때문에 가능한 한 다음 tile의 metadata를 미리 복원한다. 이는 pipeline 제안이며 아직 full kernel에서 측정하지 않았다.
3. ALU와 IMAD/FMA Heavy 사용을 함께 본다. 기존 weight dequant의 FP16 연산도 FMA 계열을 사용하므로 standalone에서 좋은 유닛 배분이 matmul에서도 최적이라고 보장하지 않는다.
4. Register 수와 spill을 다시 측정한다. 이 독립 loop의 무spill 결과가 기존 255-register kernel에도 적용되는 것은 아니다.

이 실험은 decoder 자체의 dependency와 처리량 실험이다. 제공 매핑을 실제 Sparse-Marlin packing과 연결하거나 전체 matmul 시간 개선을 검증한 결과는 아니다.

## 재현

```bash
make all sass
make run check
make profile analyze
```

6개 구현/대조군 × 4 ILP × 2 실행 종류 × 7 samples = 336개 timing sample을 저장했다. NCU report는 10개이며 CUDA Memcheck는 0 errors다. Clocks를 잠그지 않았으므로 event wall time 대신 cycle과 instruction 수를 중심으로 비교했다.

`decoder.cuh`가 구현, `experiment.cu`가 전수 검증과 benchmark, `results/summary.csv`가 timing 중앙값, `ncu_summary.csv`가 pipeline counter, `dependency_dags.json`이 probe dependency DAG다. 각 `d*_ilp*_raw.csv`에 metric의 실제 단위와 원본 값을 보관한다.

48bit 입력의 high/low 두 디코더 overlap 실험은 [PAIR_OVERLAP.md](PAIR_OVERLAP.md)를 참고한다. `make pair-run pair-profile pair-analyze`로 재현할 수 있다.
