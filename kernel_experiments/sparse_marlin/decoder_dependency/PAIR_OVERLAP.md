# 48bit metadata: high/low 디코더 overlap

RTX 3050 Laptop (sm_86), CUDA 12.6. `pair.cu`, `pair.cuh`로 별도 구현했다. 원본 Sparse-Marlin은 수정하지 않았다.

## 입력과 정렬

두 입력 레지스터를 A(낮은 주소 word), B(다음 word)라고 한다.

- T0: A=bits[0:31], B.low16=bits[32:47], B.high16=omitted.
- T1: A.low16=omitted[32:47], A.high16=bits[48:63], B=bits[64:95].

T0는 `(uint64(B)<<32 | A)<<8`, T1은 같은 64bit 값에 `>>8`을 적용한다.

| 논리 경로 | high 디코더 입력 H | low 디코더 입력 L |
|---|---|---|
| T0 | A << 8 | (B << 8) \| (A >> 24) |
| T1 | (A >> 8) \| (B << 24) | B >> 8 |

H의 high24에는 첫 8개 code, L의 low24에는 다음 8개 code가 들어간다. H.low8과 L.high8은 각 디코더가 무시한다. 64bit shift를 32bit SHF와 shift로 내려서 사용한다. 한 경로만 실행하면 정렬에 2개 명령이 필요하고, 혼합 warp에서는 두 경로의 predicated 명령 4개를 issue한다. 실제 SASS에 divergent branch는 없다.

혼합 warp는 이전 overlapping ldmatrix 배치에 맞춰 lane p와 p+16을 T0/T1로 사용했다. 레지스터 입력 이후를 측정하므로 해당 shared-memory 배치와 ldmatrix는 이번 timed loop에 없다.

## 측정 방식

`clock64`, 256 iterations x 16 unroll, 무작위 순서 7회 중앙값. 단일 warp에서는 다른 warp가 dependency를 숨기지 못한다. 두 디코더는 독립 입력으로 실행한다. 양쪽 결과를 XOR로 합쳐 다음 반복의 A에 되먹임하여, 느린 출력까지 기다린 뒤 다음 쌍이 시작되도록 했다. 따라서 수치는 정렬 + 두 디코드 + feedback XOR 1개의 시간이며, 개별 한 번의 호출을 뺄셈으로 추정한 값이 아니다. 반복 제어 비용도 소량 포함된다.

- exact: 제공한 두 함수를 그대로 사용.
- fused: 매핑/연산은 유지하고 각각 하나의 inline PTX 블록에 넣는다. 프런트엔드의 추가 마스크 변환을 억제한다. SASS에서 쌍당 LOP3가 3개 감소한다.
- serial: exact 경로에서 high 결과와 런타임 0의 AND를 low 입력에 XOR한다. 출력은 같지만 high → low 의존성이 생긴다. SASS는 추가 dependency를 LOP3 하나로 합친다. 따라서 serial과의 차이에는 이 추가 명령의 영향도 포함된다.

## 결과

한 쌍 = thread당 입력 48bit → 출력 32bit word 2개 = metadata code 16개. cycles/pair를 cycles/word와 혼동하면 안 된다.

| 구현 / 혼합 warp | 독립 쌍 1개 | 2개 | 4개 | 8개 |
|---|---:|---:|---:|---:|
| exact, 강제 직렬 대조군 | 115.07 | 68.85 | 57.86 | 63.31 |
| exact, 독립 디코더 | 70.95 | 54.47 | 54.36 | 61.78 |
| fused, 독립 디코더 | **64.45** | **50.04** | **48.46** | **48.40** |

독립 쌍 1개의 수치는 한 쌍의 완료 지연이다. 2/4/8개 열은 서로 다른 쌍까지 겹친 **쌍당 평균 처리 비용**이다. 쌍 하나가 48cycles 만에 완료된다는 뜻은 아니다. exact는 많은 쌍을 unroll했을 때 오히려 악화된다. ILP 증가만으로 처리 비용이 단조 감소한다고 가정하지 않는다.

경로별 단일 쌍 지연:

| 구현 | T0만 | T1만 | 혼합 warp |
|---|---:|---:|---:|
| 정렬 + feedback만 | 10.51 | 5.44 | 10.69 |
| exact 독립 디코더 | 71.07 | 65.19 | 70.95 |
| fused 독립 디코더 | 63.94 | 59.44 | 64.45 |

정렬 대조군의 비용은 디코더와 겹칠 수 있으므로 본 실험에서 단순히 빼면 디코더 고유 지연이 되지 않는다.

## NCU: dependency 이후의 처리량 제한

160 blocks x 128 threads, 20 SM. clock-control=none. 아래 cycles는 SM 평균 elapsed cycles이며, 행마다 독립 쌍 수에 비례하여 작업량이 증가한다. 직접 비교하려면 warp-pair 수로 정규화해야 한다.

| 구현 | 독립 쌍 | SM cycles | ALU active | FMAHeavy active | registers/thread |
|---|---:|---:|---:|---:|---:|
| exact | 1 | 1,824,357.5 | 97.44% | 25.16% | 16 |
| exact | 8 | 14,448,264 | 98.04% | 25.40% | 40 |
| fused | 1 | 1,638,706.5 | 96.49% | 28.01% | 16 |
| fused | 2 | 3,185,524.5 | 99.01% | 28.81% | 21 |
| fused | 8 | 12,645,967 | 99.57% | 29.02% | 38 |

fused는 같은 작업량에서 exact 대비 약 10.2% (1개), 12.5% (8개) cycle 감소. Fused의 정규화 처리 비용은 독립 쌍 1개일 때 12.50 SM cycles/warp-pair, 8개일 때 12.06으로 약 3.5% 감소한다. 여러 warp가 이미 dependency를 대부분 숨기고 있어, 쌍을 더 늘려도 이득은 작다. ALU가 포화되면 남은 명령량은 latency hiding으로 없앨 수 없다. 이 정규화 처리량과 단일 warp 지연 64.45cycles는 서로 다른 지표다.

모든 NCU 케이스 local load/store 0. Exhaustive 2^24 검증에서 high 함수와 두 경로의 alignment/exact/fused/serial 결과 모두 불일치 0. Omitted bits와 무시하는 bits에 garbage를 채운 상태로 검증했다. GPU memcheck 결과는 `results/pair_check.log`에 있다.

## 매핑과 적용 범위

제공한 함수를 유지했으므로 low 매핑은 [C,D,E,A,8,9,4,0], high 매핑은 [E,D,8,D,3,4,9,C]다. 두 위치의 인코더는 각 매핑에 맞춰야 한다. high가 low와 같은 매핑이라고 가정하여 변경하지 않았다. `D=3` 두 low 디코더 대조군도 구현했지만 이것은 매핑과 명령 구성이 다른 실험이므로 최종 수치로 채택하지 않았다.

이 실험은 두 word가 레지스터에 도착한 뒤의 지연을 측정한다. GMEM/SMEM load, ldmatrix latency, 실제 MMA와 weight dequant의 간섭 및 전체 matmul 시간은 포함하지 않는다. standalone에서 spill이 없다는 사실도 원본의 높은 register 사용량에서 spill이 없다는 보장은 아니다.

재현: `make pair-run pair-profile pair-analyze`, 메모리 검증 `make pair-check`.
