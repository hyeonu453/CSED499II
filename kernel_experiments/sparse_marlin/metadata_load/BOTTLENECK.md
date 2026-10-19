# Metadata load의 처리량 제한 원인

후속 실험에서 압축을 유지하며 load 명령 수를 줄이는 배치를 찾았다. [OVERLAP.md](OVERLAP.md)의 overlapping `ldmatrix.x2`는 저장 공간 192 B와 load 한 명령을 동시에 달성한다. 이 문서의 원인 분석은 기존 scalar 경로에 대한 결과다.

2026-10-19, RTX 3050 4GB Laptop GPU, sm_86, 20 SM, CUDA 12.6, NCU 2024.3.2.

## 결론

**충돌을 제거한 planar 2-word 로딩은 LSU의 load 명령 처리량에 제한된다.** 두 word를 scalar `LDS` 두 명령으로 읽는 비용이 남아, shared wavefront를 4개에서 2개로 줄여도 처리량은 그대로다.

기존 interleaved 경로는 LSU와 shared wavefront 처리율 모두 약 99%다. Planar로 바꾸면 shared wavefront 처리율은 약 50%로 내려가지만 LSU는 99%를 유지한다. 따라서 bank conflict도 기존 경로의 자원을 점유했으나, 그 비용을 제거해도 남아 있는 LSU 한계가 실행 시간을 유지한다.

## 원인을 분리한 대조 실험

기존 0/4/5번 경로에 세 가지 대조군을 추가했다.

- mode 6: planar에서 scalar `LDS` 한 번. 32-bit 값 하나를 읽는 명령 수 대조군.
- mode 7: planar에서 scalar `LDS` 네 번. 같은 두 주소를 다시 읽고 모든 결과를 사용한다. 네 명령 모두 bank conflict가 없다.
- mode 8: 각 lane의 원래 두 word를 연속된 8-byte 공간에 저장하고 `LDS.64` 한 명령으로 읽는다. Mode 5와 raw 출력이 완전히 같지만 저장 공간은 warp당 192→256 B로 증가한다. 중간 word를 복제하므로 25% metadata 저장 절감은 사라진다. 단일 명령 효과를 확인하기 위한 대조군이다.

모든 경로는 80 blocks × 256 threads × 4,096 반복, 즉 2,621,440 warp 반복을 수행한다. 초기화·최종 저장은 반복 밖에 있다. 반복 중 global 접근, decoder, MMA는 없다. Shared 할당은 모두 8 KiB로 동일하다. NCU 프로파일링과 별도로 9회씩 순서를 무작위화한 timing을 수집했다.

## NCU 결과

| 경로 | 반복당 load 명령 | 반복당 shared wavefront | NCU SM cycles | ldmatrix 대비 | LSU %peak | Shared wavefront %peak |
|---|---:|---:|---:|---:|---:|---:|
| ldmatrix.x2, mode 0 | 1 | 2 | 267,932 | 1.000 | 98.42% | 98.02% |
| Interleaved scalar×2, mode 4 | 2 | 4 | 529,554.5 | 1.976 | 99.20% | 99.06% |
| Planar scalar×2, mode 5 | 2 | 2 | 529,575 | 1.976 | 99.20% | 49.56% |
| Planar scalar×1, mode 6 | 1 | 1 | 266,344 | 0.994 | 98.81% | 49.32% |
| Planar scalar×4, mode 7 | 4 | 4 | 1,056,315.5 | 3.942 | 99.37% | 49.66% |
| Padded LDS.64, mode 8 | 1 | 2 | 268,942.5 | 1.004 | 98.05% | 97.65% |

%peak는 NVIDIA hardware counter의 peak 대비 처리율이다. Shared 열은 byte 대역폭이 아니라 wavefront 처리율이다.

측정 metric:

- `sm__cycles_elapsed.avg`: 표의 SM cycles. SM별 elapsed cycle의 평균이며, 클록이 다른 run의 wall time 대신 비교했다.
- `sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed`: LSU 명령 처리율.
- `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed`: shared wavefront 처리율.
- `derived__memory_l1_wavefronts_shared_excessive`: 초과 wavefront. Mode 4만 5,242,880개이고 다른 표의 경로는 0개다.

모든 load 횟수는 NCU dynamic instruction count로 검증했다. Scalar×1/×2/×4는 각각 2,621,440 / 5,242,880 / 10,485,760개다. Mode 8도 실제 SASS가 `LDS.64`이고 dynamic load count가 2,621,440개임을 확인했다.

## 인과관계 해석

1. **Bank conflict 제거만으로는 시간 변화가 없다.** Mode 4→5에서 초과 wavefront가 0이 되고 shared 처리율은 반으로 내려가지만, load 명령 수와 실행 cycles는 같다.
2. **Bank conflict 없이 명령 수만 늘리면 비용이 비례한다.** Mode 6→5→7에서 1→2→4개 scalar load에 대해 cycles도 약 1→2→4배다. LSU는 모두 약 99%다.
3. **같은 두 word를 한 명령으로 읽으면 처리량이 회복된다.** Mode 5→8에서 raw 출력은 같고 load 명령 수는 절반이다. Cycles는 529,575→268,942.5로 약 49% 감소한다. Mode 8은 오히려 더 큰 입력 저장 공간을 사용하므로, 저장량 감소로 설명되는 결과가 아니다.
4. **ALU·register spill이 주원인이라는 근거가 없다.** Mode 5의 ALU cycle 활성도는 21.73%, register 수는 22/thread로 baseline과 같다. 표의 모든 경로에서 local load/store spill은 0개다. 반복 중 DRAM 접근도 없다.

NCU의 MIO throttle sampling도 이를 뒷받침한다. Mode 5의 MIO throttle은 5,426 / 10,289 samples로 약 52.7%이며, long scoreboard는 15 samples에 그친다. 다만 stall sampling 하나만으로 결론내리지 않고 위 명령 수 대조 실험과 함께 해석했다.

이 테스트 조건에서 실측 load 명령 수를 SM 수와 SM cycles로 나누면 약 0.49~0.50 warp-load 명령/SM cycle이다. 이는 **이번 workload에서 관측한 유효 처리율**이며 모든 NVIDIA GPU의 명령 처리율에 대한 주장으로 사용하지 않는다.

## 실제 Sparse-Marlin에 대한 의미

압축 metadata를 읽을 때 bank conflict를 없애는 것과 load 명령 수를 줄이는 것은 별개의 최적화다. 이번 raw 실험에서는 전자는 성공했으나 후자의 비용이 남았다.

현재 planar 배치의 두 word는 64 B 떨어져 있으므로 그대로 `LDS.64`로 합칠 수 없다. 두 word를 연속·8-byte 정렬 배치하면 한 명령으로 읽을 수 있지만, mode 8처럼 중복 저장을 넣으면 압축 이득이 사라진다. 압축 크기를 유지하는 다른 cooperative load/재분배 방법은 load 명령 수뿐 아니라 shuffle 명령과 register 비용까지 다시 측정해야 한다.

이 결론은 독립 shared-load 루프의 병목에 대한 것이다. 실제 Marlin은 GMEM 비동기 복사, weight dequantization, MMA를 겹치므로 동일한 slowdown을 적용할 수 없다. 압축 경로의 전체 커널 성능은 아직 측정하지 않았다.

## 검증 및 재현

모든 raw 경로와 대조군을 4 stages × 32 lanes에서 CPU 기준값과 비교했다. Mode 6은 첫 word를 두 번 반환하고, mode 7은 각 word를 두 번 더한 값으로 검증한다. Memcheck와 synccheck는 모두 0 errors다. Timing은 9 modes × 2 kinds × 9 samples = 162개다.

```bash
make all sass RESULTS=results/bottleneck
make run check profile analyze RESULTS=results/bottleneck MODES="0 4 5 6 7 8"
```

원본은 `experiment.cu`, SASS·timing·NCU CSV·sanitizer log는 `results/bottleneck/`에 있다. Clocks는 잠그지 않았으므로 event 시간이나 NCU wall time의 run 간 비교 대신 cycles와 명령 횟수를 중심으로 해석했다.

참고: [NVIDIA Profiling Guide의 LSU 정의](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#pipelines), [명령 처리 압력과 메모리 처리량을 구분하는 Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/).
