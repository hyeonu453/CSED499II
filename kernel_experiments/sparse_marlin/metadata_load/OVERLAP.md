# 압축을 유지하는 overlapping ldmatrix.x2

## 결과

**Warp당 metadata 저장 공간 192 B를 유지하면서 scalar load 두 개를 `ldmatrix.x2` 한 개로 대체했다.** Padding, 중복 저장, lane shuffle 없이 가능하다. 로딩만의 처리량은 기존 256 B metadata의 `ldmatrix.x2`와 같아졌다. 복원한 fragment는 기존 각 lane의 64-bit metadata와 일치한다.

## 배치

인접 lane을 묶는 대신 lane p와 lane p+16을 묶는다(p=0..15). 각 lane의 16개 metadata를 각각 48-bit로 압축하고, 두 lane의 code 32개를 96-bit로 연결한다. 이를 X[p],Y[p],Z[p]라는 세 개의 32-bit word로 나눈다.

```text
X[p]: lane p의 첫 32-bit
Y[p]: lane p의 나머지 16-bit + lane p+16의 첫 16-bit
Z[p]: lane p+16의 나머지 32-bit
```

Shared에 다음처럼 plane별로 저장한다.

```text
word  0..15: X0 X1 ... X15
word 16..31: Y0 Y1 ... Y15
word 32..47: Z0 Z1 ... Z15
```

`ldmatrix.x2`에 지정하는 두 8×8×16-bit 행렬은 다음과 같다.

```text
첫 번째 행렬: word  0..31 = X + Y   (128 B)
두 번째 행렬: word 16..47 = Y + Z   (128 B)
                         ↑ Y 부분을 겹쳐 읽는다
```

두 행렬은 shared memory의 같은 Y 공간을 참조한다. 두 번 읽어도 저장은 한 번이며, 총 공간은 48 words=192 B다. 행 시작 주소는 모두 16-byte 정렬이다.

행 주소 제공은 다음 코드로 구현했다. Lane 16..31의 주소는 유효한 주소를 반복 제공하며, `.x2`의 행 주소는 lane 0..15에서 취한다.

```cpp
int row_word = (lane & 7) * 4 + ((lane >> 3) & 1) * 16;
ldmatrix.x2(stage + row_word);
```

실제 결과는 lane t가 `stage[t]`, `stage[t+16]`을 받는다. 따라서:

| Lane | 첫 register | 둘째 register | 복원할 구간 |
|---|---|---|---|
| p (0..15) | X[p] | Y[p] | 연결한 code의 하위 48-bit |
| p+16 | Y[p] | Z[p] | 연결한 code의 상위 48-bit |

각 lane이 자신의 결과를 직접 받으므로 shuffle이 필요 없다. 기존 helper의 두 register 인터페이스도 유지한다. 두 개의 24-bit → 32-bit SWAR decoder로 원래 두 metadata register를 만든다.

이 실험은 CPU에서 새로운 global packing을 미리 생성하고 동일한 순서로 shared에 복사한다. 실제 Marlin에 적용하려면 packing, metadata global/shared stride와 warp별 배치를 함께 변경해야 한다.

## GPU 검증과 측정

2026-10-19, RTX 3050 4GB Laptop GPU (sm_86, 20 SM), CUDA 12.6, NCU 2024.3.2. 80 blocks × 256 threads × 4,096 반복, 2,621,440 warp iterations. Clocks는 잠그지 않았으므로 run 간 비교는 NCU SM cycles를 사용했다.

| 경로 | Warp당 저장 공간 | Load 명령/반복 | NCU SM cycles | 기존 대비 | LSU %peak | ALU cycle 활성도 |
|---|---:|---:|---:|---:|---:|---:|
| 기존 4-bit ldmatrix.x2 | 256 B | 1 | 268,100.5 | 1.000 | 98.35% | 36.85% |
| 인접 lane planar scalar×2 | 192 B | 2 | 530,619.5 | 1.979 | 99.00% | 21.69% |
| Overlap ldmatrix, raw | 192 B | 1 | 267,710 | 0.999 | 98.31% | 43.00% |
| Overlap ldmatrix + 복원 | 192 B | 1 | 1,710,979 | 6.382 | 15.38% | 98.66% |

Raw overlap과 full decode 모두 `smsp__inst_executed_op_ldsm.sum=2,621,440`으로 한 번의 load가 반복마다 실행됨을 검증했다. Scalar shared load, 초과 shared wavefront, local spill은 없다. Raw overlap register 수는 기존과 동일한 22/thread이고 복원 포함 시 32/thread다.

여기서 LSU의 **명령 수/논리 작업량**은 줄었지만 raw 루프의 LSU %peak는 여전히 약 98%다. 같은 명령 처리 자원으로 같은 양의 작업을 더 빨리 끝내기 때문이다. "부하를 줄인다"를 곧바로 utilization % 감소로 해석하면 안 된다.

Full decode에서는 LSU보다 **ALU 복원 비용**이 병목이 되었다. 이 실험만으로 전체 matmul이 빨라진다고 결론낼 수 없다. 또한 unique 저장 공간은 192 B지만, `ldmatrix.x2`는 Y를 중복해서 읽으므로 nominal shared read payload는 256 B다. Shared read량까지 25% 줄인 결과가 아니다.

복원 결과는 4 stages × 32 lanes × 16 codes에서 기존 reference와 bit-exact로 일치한다. Raw 결과는 `stage[t]`, `stage[t+16]` CPU reference와 일치한다. Memcheck와 synccheck는 모두 0 errors다. 모든 11개 mode에 대해 dependent/throughput 각각 9회, 총 198개 timing sample을 수집했다. 무작위 code 기반의 결정적 검증이며 전수 검증은 아니다.

## 재현

```bash
make all sass RESULTS=results/overlap
make run check profile analyze RESULTS=results/overlap MODES="0 5 9 10"
```

Mode 9는 raw, mode 10은 복원까지 포함한다. 소스는 `experiment.cu`, raw CSV·NCU counter·SASS·sanitizer log는 `results/overlap/`에 있다. 아직 실제 Sparse-Marlin 커널에 통합하지 않았다.
