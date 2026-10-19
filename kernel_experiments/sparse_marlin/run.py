#!/usr/bin/env python3
"""Check and time the extension built from this directory's Sparse-Marlin copy."""

import argparse
import json
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "bench"])
    parser.add_argument("--m", type=int, default=16, help="batch size")
    parser.add_argument("--n", type=int, default=512, help="output features, multiple of 256")
    parser.add_argument("--k", type=int, default=1024, help="input features, multiple of 128")
    parser.add_argument("--groupsize", type=int, choices=[-1, 128], default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.m < 1 or args.n < 256 or args.n % 256 or args.k < 128 or args.k % 128:
        parser.error("m >= 1, n must be a positive multiple of 256, k of 128")
    if args.warmup < 1 or args.iters < 1:
        parser.error("warmup and iters must be positive")

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        parser.exit(1, "CUDA GPU is unavailable; run on a GPU-enabled host.\n")

    # Prefer this source tree over any installed kernel_bench extension.
    sys.path.insert(0, str(SRC))
    import marlin
    import marlin_cuda
    from test_2_4 import gen_quant4_NT

    for module in (marlin, marlin_cuda):
        if not Path(module.__file__).resolve().is_relative_to(SRC):
            parser.exit(1, "Local extension is missing; run make build first.\n")
    print(f"extension: {marlin_cuda.__file__}", file=sys.stderr)
    np.random.seed(0)
    torch.manual_seed(0)
    # Small cases exercise both supported scale layouts and several batch sizes.
    cases = [(m, 512, 1024, g) for g in (-1, 128) for m in (1, 16, 32)]
    requested = (args.m, args.n, args.k, args.groupsize)
    if args.mode == "bench":
        cases = [requested]
    elif requested not in cases:
        cases.append(requested)

    with torch.inference_mode():
        for m, n, k, groupsize in cases:
            a = torch.randn((m, k), device="cuda", dtype=torch.float16)
            ref_b, b, scales, meta = gen_quant4_NT(n, k, groupsize)
            c = torch.empty((m, n), device="cuda", dtype=torch.float16)
            workspace = torch.zeros(n // 128 * 16, device="cuda", dtype=torch.int32)

            def invoke():
                marlin.mul_2_4(a, b, meta, c, scales, workspace)

            invoke()
            reference = a @ ref_b
            error = ((c - reference).abs().mean() / reference.abs().mean()).item()
            if not error < 0.002:
                raise AssertionError(f"m={m} n={n} k={k} group={groupsize}: relative MAE={error}")
            result = dict(m=m, n=n, k=k, groupsize=groupsize, relative_mae=error)
            if args.mode == "bench":
                for _ in range(args.warmup):
                    invoke()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iters):
                    invoke()
                end.record()
                end.synchronize()
                result["mean_us"] = start.elapsed_time(end) * 1000 / args.iters
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
