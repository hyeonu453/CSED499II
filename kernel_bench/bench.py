import os
import argparse
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

import torch
import time

dev = torch.device("cuda:0")
sms = torch.cuda.get_device_properties(dev).multi_processor_count

import marlin
import exllamav2_ext
import macko_spmv


# Throughput uses dense-equivalent FLOPs and nominal tensor bytes, not DRAM counters.
def benchmark(f, warmup=1, iter=10, profile=False):
    """Average CUDA-event seconds per call; use profile=True with ncu --profile-from-start off."""
    if warmup < 0 or iter <= 0:
        raise ValueError("warmup must be nonnegative and iter must be positive")
    for i in range(warmup):
        f()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Exclude preprocessing and warmup from Nsight Compute's collection range.
    if profile:
        torch.cuda.profiler.start()
    try:
        start.record()
        for i in range(iter):
            f()
        end.record()
        torch.cuda.synchronize()
    finally:
        if profile:
            torch.cuda.profiler.stop()
    # This interval includes launch gaps and profiler overhead; under ncu,
    # use gpu__time_duration.sum for the actual kernel execution time.
    res = start.elapsed_time(end) / iter / 1000
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.0)
    return res

def get_marlin_prob(m, n, k, groupsize=-1):
    if groupsize == -1:
        groupsize = k
    dev = torch.device("cuda:0")
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B = torch.randint(low=-(2**31), high=2**31, size=(k * n // 8,), dtype=torch.int32, device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B, C, B_ref, s

def get_exl2_prob(m, n, k, bits=4, groupsize=-1):
    """Return A, EXL2 tensor dict, C, decoded B_ref, and packed EXL2 scales."""
    assert bits in [2, 3, 4, 5, 6]
    if groupsize == -1:
        groupsize = k
    assert m > 0 and n > 0 and k > 0
    assert n % 32 == 0 and k % 32 == 0
    assert groupsize > 0 and groupsize % 32 == 0 and k % groupsize == 0
    # EXL2 stores permutation, group offsets and group lengths as uint16.
    assert k <= 65535 and k * bits // 32 <= 65535
    dev = torch.device("cuda:0")
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    groups = k // groupsize
    q_groups = torch.empty((groups, 2), dtype=torch.int32)
    q_groups[:, 0] = bits
    q_groups[:, 1] = torch.arange(groups) * (groupsize * bits // 32)
    B = {
        "q_weight": torch.randint(-(2**31), 2**31, (k * bits // 32, n), dtype=torch.int32, device=dev),
        "q_perm": torch.arange(k, device=dev).short(),
        "q_invperm": torch.arange(k, device=dev).short(),
        "q_groups": q_groups.to(device=dev, dtype=torch.int16).flatten(),
        "q_group_map": exllamav2_ext.make_group_map(q_groups.short().flatten(), k * bits // 32).to(dev),
        # Packed scale 15 decodes to (15 + 1)**2 * (1 / 256) = 1.
        "q_scale_max": torch.full((groups,), 1 / 256, dtype=torch.half, device=dev),
        "temp_dq": torch.empty((0,), dtype=torch.half, device=dev),
    }
    s = torch.full((groups, n // 8), -1, dtype=torch.int32, device=dev)
    B_ref = torch.empty((k, n), dtype=torch.half, device=dev)
    handle, q_weight = _make_exl2_matrix(B, s)
    try:
        exllamav2_ext.reconstruct(handle, B_ref)
        torch.cuda.synchronize()
    finally:
        exllamav2_ext.free_q_matrix(handle)
    # Same logical shapes and tuple order as get_marlin_prob; EXL2 needs metadata.
    return A, B, C, B_ref, s


def _make_exl2_matrix(B, s):
    none = torch.empty((0,), device="meta")
    # make_q_matrix shuffles weights in place. Keep the original reusable.
    q_weight = B["q_weight"].clone()
    handle = exllamav2_ext.make_q_matrix(
        q_weight, B["q_perm"], B["q_invperm"], s, B["q_scale_max"],
        B["q_groups"], B["q_group_map"], none, none, none, none,
        B["temp_dq"], 0,
    )
    return handle, q_weight

def gen_2_4(m, n, k, dev):
    B = torch.randint(low=-(2**31), high=2**31, size=(k * n // 8 // 2,), dtype=torch.int32, device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    meta = torch.ones((n * k // 16,), dtype=torch.int16, device=dev) * (-4370)
    return B, B_ref, meta

def get_problem_24(m, n, k, groupsize=-1):
    dev = torch.device("cuda:0")
    B, B_ref, meta = gen_2_4(m, n, k, dev)
    if groupsize == -1:
        s = torch.zeros((1, n), dtype=torch.half, device=dev)
    else:
        s = torch.zeros(
            (((k // 2) // (groupsize // 2)), n), dtype=torch.half, device=dev
        )
    torch.cuda.synchronize()
    return B, B_ref, s, meta

def get_spmv_prob(m, n, k, density=0.5):
    assert m == 1
    assert 0 <= density <= 1
    dev = torch.device("cuda:0")
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B_ref = torch.rand((k, n), dtype=torch.half, device=dev)
    B_ref[torch.rand_like(B_ref) > density] = 0
    # Sparse kernels compute B_ref.T @ A.T, equivalent to (A @ B_ref).T.
    B_t = B_ref.t().contiguous()
    B_csr = B_t.to_sparse_csr()
    B_macko = macko_spmv.compress(B_t)
    torch.cuda.synchronize()
    return A, B_macko, B_csr, B_ref

def get_structure_spmv_prob(m, n, k):
    """Return A(1,k), MACKO/CSR of B_ref.T(n,k), and dense B_ref(k,n)."""
    assert m == 1
    assert n > 0 and k > 0 and n % 16 == 0 and k % 32 == 0
    dev = torch.device("cuda:0")
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    # Two nonzeros per four consecutive k entries in every output channel.
    B_t = torch.zeros((n, k // 4, 4), dtype=torch.half, device=dev)
    indices = torch.rand((n, k // 4, 4), device=dev).argsort(dim=-1)[..., :2]
    values = torch.rand((n, k // 4, 2), dtype=torch.half, device=dev) + 0.5
    B_t.scatter_(-1, indices, values)
    B_t = B_t.view(n, k)
    B_ref = B_t.t().contiguous()
    B_csr = B_t.to_sparse_csr()
    B_macko = macko_spmv.compress(B_t)
    torch.cuda.synchronize()
    return A, B_macko, B_csr, B_ref


def benchmark_sparse_marlin(A, B, meta, C, s, warmup=1, iter=10, profile=False):
    workspace = torch.zeros(
        C.shape[1] // 128 * 16, device=torch.device("cuda:0"), dtype=torch.int32
    )

    f = lambda: marlin.mul_2_4(A, B, meta, C, s, workspace, -1, -1, sms)
    res = benchmark(f, warmup, iter, profile=profile)

    return {
        "s": res,
        "TFLOP/s": 2 * A.numel() * C.shape[1] / res / 10**12,
        "GB/s": (2 * A.numel() + 4 * B.numel() + 2 * meta.numel() + 2 * C.numel() + 2 * s.numel())
        / res
        / 10**9,
    }

def benchmark_dense_marlin(A, B, C, s, warmup=1, iter=10, profile=False):
    workspace = torch.zeros(
        C.shape[1] // 128 * 16, device=torch.device("cuda:0"), dtype=torch.int32
    )

    f = lambda: marlin.mul(A, B, C, s, workspace, -1, -1, sms)
    res = benchmark(f, warmup, iter, profile=profile)

    return {
        "s": res,
        "TFLOP/s": 2 * A.numel() * C.shape[1] / res / 10**12,
        "GB/s": (2 * A.numel() + 4 * B.numel() + 2 * C.numel() + 2 * s.numel())
        / res
        / 10**9,
    }

def benchmark_exl2(A, B, C, s, warmup=1, iter=10, profile=False):
    handle, q_weight = _make_exl2_matrix(B, s)
    # force_cuda=True prevents reconstruction + cuBLAS fallback for large m.
    f = lambda: exllamav2_ext.gemm_half_q_half(A, handle, C, True)
    try:
        # q_gemm_autotune.cuh uses AT_NUM_MEASURE=200, then selects on call 201.
        # Complete internal autotuning before the profiling/timing range.
        for i in range(201):
            f()
        res = benchmark(f, warmup, iter, profile=profile)
    finally:
        torch.cuda.synchronize()
        exllamav2_ext.free_q_matrix(handle)
    size = sum(t.numel() * t.element_size() for t in (A, C, s, *B.values()))
    return {
        "s": res,
        "TFLOP/s": 2 * A.numel() * C.shape[1] / res / 10**12,
        "GB/s": size / res / 10**9,
    }

def benchmark_macko(A, B, warmup=1, iter=10, profile=False):
    assert A.shape == (1, B[4])
    V = A.flatten().contiguous()
    f = lambda: macko_spmv.multiply(B, V)
    res = benchmark(f, warmup, iter, profile=profile)
    size = sum(t.numel() * t.element_size() for t in B[:3])
    return {
        "s": res,
        "TFLOP/s": 2 * A.numel() * B[3] / res / 10**12,
        "GB/s": (size + 2 * A.numel() + 2 * B[3]) / res / 10**9,
    }

def benchmark_cublas(A, B, C, warmup=1, iter=10, profile=False):
    A = A.contiguous()
    B = B.contiguous()
    assert C.is_contiguous()
    f = lambda: torch.mm(A, B, out=C)
    res = benchmark(f, warmup, iter, profile=profile)
    return {
        "s": res,
        "TFLOP/s": 2 * A.numel() * C.shape[1] / res / 10**12,
        "GB/s": (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / res / 10**9,
    }

def benchmark_cusparse(A, B, warmup=1, iter=10, profile=False):
    # B is CSR of shape (n, k), as returned by both SpMV generators.
    assert B.layout == torch.sparse_csr and A.shape == (1, B.shape[1])
    V = A.flatten().contiguous()
    C = torch.empty((B.shape[0],), dtype=A.dtype, device=A.device)
    f = lambda: torch.addmv(C, B, V, beta=0, out=C)
    res = benchmark(f, warmup, iter, profile=profile)
    size = sum(t.numel() * t.element_size() for t in (B.values(), B.crow_indices(), B.col_indices()))
    return {
        "s": res,
        "TFLOP/s": 2 * A.numel() * B.shape[0] / res / 10**12,
        "GB/s": (size + 2 * A.numel() + 2 * C.numel()) / res / 10**9,
    }

def benchmark_cusparselt(A, B, warmup=1, iter=10, profile=False):
    # B is the dense (k, n) B_ref, with 2:4 sparsity along k.
    assert A.shape[0] > 0 and (A.shape[0] < 8 or A.shape[0] % 8 == 0)
    assert A.shape[1] == B.shape[0]
    assert A.dtype == B.dtype == torch.half
    m = A.shape[0]
    padded_m = max(8, m)
    k, n = B.shape
    assert n % 16 == 0 and k % 32 == 0
    B_t = B.t().contiguous()
    if not ((B_t.view(n, k // 4, 4) != 0).sum(dim=-1) == 2).all().item():
        raise ValueError("cuSPARSELt requires exactly two nonzeros per four k entries")
    B_compressed = torch._cslt_compress(B_t)
    # FP16 dense dimensions must be multiples of 8; pad only when m < 8.
    if m < 8:
        A_padded = torch.zeros((padded_m, k), dtype=A.dtype, device=A.device)
        A_padded[:m].copy_(A)
    else:
        A_padded = A.contiguous()
    V = A_padded.t()
    f = lambda: torch._cslt_sparse_mm(B_compressed, V)
    res = benchmark(f, warmup, iter, profile=profile)
    return {
        "s": res,
        # Effective throughput of the original problem, not padded work.
        "TFLOP/s": 2 * A.numel() * n / res / 10**12,
        "GB/s": (2 * A_padded.numel() + B_compressed.numel() * B_compressed.element_size() + 2 * padded_m * n) / res / 10**9,
        "padded_m": padded_m,
    }

MODELS = {
    "Llama7B": [(4096, 3 * 4096), (4096, 4096), (4096, 2 * 10752), (10752, 4096)],
    "Llama13B": [(5120, 3 * 5120), (5120, 5120), (5120, 2 * 13568), (13568, 5120)],
    "Llama33B": [(6656, 3 * 6656), (6656, 6656), (6656, 2 * 17664), (17664, 6656)],
    # "Llama65B": [(8192, 3 * 8192), (8192, 8192), (8192, 2 * 21760), (21760, 8192)],
    # "Falcon180B": [
    #     # Note that parallel attention and FC allows layer fusions
    #     (14848, 14848 * 5 + 1024),
    #     (14848 * 5, 14848),
    # ],
}

ALL = True

def run_sweep(models, batchsizes, run_case):
    for model, layers in models.items():
        print(model)
        for batch in batchsizes:
            totals = {}
            for k, n in layers:
                results = run_case(batch, n, k)
                baseline = next(iter(results.values()))["s"]
                for name, res in results.items():
                    total = totals.setdefault(name, {"s": 0, "TFLOP/s": 0, "GB/s": 0, "speedup": 0})
                    total["s"] += res["s"]
                    total["TFLOP/s"] += res["TFLOP/s"] * res["s"]
                    total["GB/s"] += res["GB/s"] * res["s"]
                    total["speedup"] += baseline
            for name, total in totals.items():
                for key in ("TFLOP/s", "GB/s", "speedup"):
                    total[key] /= total["s"]
                print(
                    "[%s] batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f, speedup=%.2f"
                    % (name, batch, total["s"], total["TFLOP/s"], total["GB/s"], total["speedup"])
                )
        print()


def benchmark_marlin_kernels(models, batchsizes, warmup, iter, profile):
    # cuBLAS + Marlin + cuSPARSELt + Sparse-Marlin; speedup relative to cuBLAS.
    def run_case(m, n, k):
        A, B, C, B_ref, s = get_marlin_prob(m, n, k, groupsize=128)
        results = {"cuBLAS": benchmark_cublas(A, B_ref, C, warmup, iter, profile)}
        results["NN"] = benchmark_dense_marlin(A, B, C, s, warmup, iter, profile)
        del B, B_ref, s
        # get_problem_24's B_ref is synthetic dense data, not a 2:4 matrix.
        B_ref = torch.rand((k // 4, 4, n), dtype=torch.half, device=A.device) + 0.5
        B_ref[:, :2] = 0
        results["cuSPARSELt"] = benchmark_cusparselt(A, B_ref.view(k, n), warmup, iter, profile)
        del B_ref
        B, B_ref, s, meta = get_problem_24(m, n, k, groupsize=128)
        results["2:4"] = benchmark_sparse_marlin(A, B, meta, C, s, warmup, iter, profile)
        return results

    print("groupsize=128\n")
    run_sweep(models, batchsizes, run_case)


def benchmark_exl2_kernels(models, batchsizes, bits=[2, 3, 4, 5], warmup=1, iter=10, profile=False):
    # EXL2 bit-width sweep; speedup relative to cuBLAS.
    if not bits or any(bit not in (2, 3, 4, 5, 6) for bit in bits):
        raise ValueError("bits must be a nonempty sequence of supported EXL2 bit widths")

    def run_case(m, n, k):
        results = {}
        for bit in bits:
            A, B, C, B_ref, s = get_exl2_prob(m, n, k, bits=bit, groupsize=128)
            if not results:
                results["cuBLAS"] = benchmark_cublas(A, B_ref, C, warmup, iter, profile)
            results["EXL2-%dbit" % bit] = benchmark_exl2(A, B, C, s, warmup, iter, profile)
            del A, B, C, B_ref, s
        return results

    print("groupsize=128\n")
    run_sweep(models, batchsizes, run_case)


def benchmark_dense_kernels(models, batchsizes, warmup=1, iter=10, profile=False):
    # cuBLAS vs Marlin; speedup relative to cuBLAS.
    def run_case(m, n, k):
        A, B, C, B_ref, s = get_marlin_prob(m, n, k, groupsize=128)
        return {
            "cuBLAS": benchmark_cublas(A, B_ref, C, warmup, iter, profile),
            "NN": benchmark_dense_marlin(A, B, C, s, warmup, iter, profile),
        }

    print("groupsize=128\n")
    run_sweep(models, batchsizes, run_case)


def benchmark_sparse_kernels(models, batchsizes, warmup=1, iter=10, profile=False):
    # cuSPARSELt vs Sparse-Marlin; speedup relative to cuSPARSELt.
    def run_case(m, n, k):
        A = torch.randn((m, k), dtype=torch.half, device=dev)
        C = torch.zeros((m, n), dtype=torch.half, device=dev)
        B_ref = torch.rand((k // 4, 4, n), dtype=torch.half, device=dev) + 0.5
        B_ref[:, :2] = 0
        results = {"cuSPARSELt": benchmark_cusparselt(A, B_ref.view(k, n), warmup, iter, profile)}
        del B_ref
        B, B_ref, s, meta = get_problem_24(m, n, k, groupsize=128)
        results["2:4"] = benchmark_sparse_marlin(A, B, meta, C, s, warmup, iter, profile)
        return results

    print("groupsize=128\n")
    run_sweep(models, batchsizes, run_case)


def benchmark_quant_kernels(models, batchsizes, warmup=1, iter=10, profile=False):
    # Marlin vs EXL2 4bit; speedup relative to Marlin.
    def run_case(m, n, k):
        A, B, C, B_ref, s = get_marlin_prob(m, n, k, groupsize=128)
        results = {"NN": benchmark_dense_marlin(A, B, C, s, warmup, iter, profile)}
        del A, B, C, B_ref, s
        A, B, C, B_ref, s = get_exl2_prob(m, n, k, bits=4, groupsize=128)
        results["EXL2-4bit"] = benchmark_exl2(A, B, C, s, warmup, iter, profile)
        return results

    print("groupsize=128\n")
    run_sweep(models, batchsizes, run_case)


def benchmark_spmv_kernels(models, densities, warmup=1, iter=10, profile=False):
    # MACKO vs cuSPARSE on the same matrix; speedup relative to cuSPARSE.
    for density in densities:
        assert 0 <= density <= 1
        print("density=%.3f\n" % density)

        def run_case(m, n, k):
            A, B_macko, B_csr, B_ref = get_spmv_prob(m, n, k, density)
            return {
                "cuSPARSE": benchmark_cusparse(A, B_csr, warmup, iter, profile),
                "MACKO": benchmark_macko(A, B_macko, warmup, iter, profile),
            }

        run_sweep(models, [1], run_case)


def benchmark_structured_spmv_kernels(models, densities, warmup=1, iter=10, profile=False):
    # All three kernels share one 2:4 matrix; speedup relative to cuSPARSE.
    densities = list(densities)
    if any(density != 0.5 for density in densities):
        raise ValueError("Structured 2:4 SpMV requires density=0.5")
    for density in densities:
        print("density=%.3f (2:4)\n" % density)

        def run_case(m, n, k):
            A, B_macko, B_csr, B_ref = get_structure_spmv_prob(m, n, k)
            return {
                "cuSPARSE": benchmark_cusparse(A, B_csr, warmup, iter, profile),
                "MACKO": benchmark_macko(A, B_macko, warmup, iter, profile),
                "cuSPARSELt": benchmark_cusparselt(A, B_ref, warmup, iter, profile),
            }

        run_sweep(models, [1], run_case)


def benchmark_kernel(kernel, m, n, k, **kwargs):
    groupsize = kwargs.get("groupsize", 128)
    warmup = kwargs.get("warmup", 1)
    iter = kwargs.get("iter", 10)
    dev = torch.device("cuda:0")
    if kernel == 'marlin':
        A, B, C, B_ref, s = get_marlin_prob(m, n, k, groupsize)
        res = benchmark_dense_marlin(A, B, C, s, warmup, iter, profile=True)
    elif kernel == 'sparse-marlin':
        A = torch.randn((m, k), dtype=torch.half, device=dev)
        C = torch.zeros((m, n), dtype=torch.half, device=dev)
        B_24, B_ref, s_24, meta = get_problem_24(m, n, k, groupsize)
        res = benchmark_sparse_marlin(A, B_24, meta, C, s_24, warmup, iter, profile=True)
    elif kernel == 'exl2':
        bits = kwargs.get("bits", 4)
        A, B, C, B_ref, s = get_exl2_prob(m, n, k, bits, groupsize)
        res = benchmark_exl2(A, B, C, s, warmup, iter, profile=True)
    elif kernel == 'macko':
        assert m == 1
        density = kwargs.get("density", 0.5)
        assert 0 <= density <= 1
        A = torch.randn((m, k), dtype=torch.half, device=dev)
        B_ref = torch.rand((k, n), dtype=torch.half, device=dev)
        B_ref[torch.rand_like(B_ref) > density] = 0
        B = macko_spmv.compress(B_ref.t().contiguous())
        res = benchmark_macko(A, B, warmup, iter, profile=True)
    elif kernel == 'cublas':
        A = torch.randn((m, k), dtype=torch.half, device=dev)
        B = torch.randn((k, n), dtype=torch.half, device=dev)
        C = torch.zeros((m, n), dtype=torch.half, device=dev)
        res = benchmark_cublas(A, B, C, warmup, iter, profile=True)
    elif kernel == 'cusparselt':
        assert m > 0 and (m < 8 or m % 8 == 0)
        assert n > 0 and k > 0 and n % 16 == 0 and k % 32 == 0
        A = torch.randn((m, k), dtype=torch.half, device=dev)
        # Build only the dense 2:4 weights; the benchmark handles Lt compression.
        B_t = torch.zeros((n, k // 4, 4), dtype=torch.half, device=dev)
        indices = torch.rand((n, k // 4, 4), device=dev).argsort(dim=-1)[..., :2]
        values = torch.rand((n, k // 4, 2), dtype=torch.half, device=dev) + 0.5
        B_t.scatter_(-1, indices, values)
        B = B_t.view(n, k).t().contiguous()
        res = benchmark_cusparselt(A, B, warmup, iter, profile=True)
    elif kernel == 'cusparse':
        assert m == 1
        density = kwargs.get("density", 0.5)
        assert 0 <= density <= 1
        A = torch.randn((m, k), dtype=torch.half, device=dev)
        B_ref = torch.rand((k, n), dtype=torch.half, device=dev)
        B_ref[torch.rand_like(B_ref) > density] = 0
        B = B_ref.t().contiguous().to_sparse_csr()
        res = benchmark_cusparse(A, B, warmup, iter, profile=True)
    else:
        raise ValueError(f"Unsupported Kernel: {kernel}")
    return res

def parse_args():
    parser = argparse.ArgumentParser(description="Kernel performance benchmarks")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    compare = subparsers.add_parser("compare", help="Compare kernels across model layers")
    compare.add_argument("comparison", choices=["dense", "sparse", "quant", "spmv", "structured-spmv", "marlin", "exl2"])
    compare.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    compare.add_argument("--batchsizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256] + ([512, 1024, 2048] if ALL else []))
    compare.add_argument("--densities", type=float, nargs="+", default=[0.5], help="Densities for SpMV comparisons; structured-spmv requires 0.5")
    compare.add_argument("--bits", type=int, nargs="+", choices=[2, 3, 4, 5, 6], default=[2, 3, 4, 5], help="Bit widths for the EXL2 comparison")
    compare.add_argument("--warmup", type=int, default=1)
    compare.add_argument("--iter", type=int, default=10)
    compare.add_argument("--profile", action="store_true", help="Enable NCU ranges; run ncu with --profile-from-start off")

    single = subparsers.add_parser("single", help="Profile one kernel with ncu --profile-from-start off")
    single.add_argument("kernel", choices=["marlin", "sparse-marlin", "exl2", "macko", "cublas", "cusparselt", "cusparse"])
    single.add_argument("-m", type=int, required=True)
    single.add_argument("-n", type=int, required=True)
    single.add_argument("-k", type=int, required=True)
    single.add_argument("--bits", type=int, choices=[2, 3, 4, 5, 6], default=4)
    single.add_argument("--groupsize", type=int, default=128)
    single.add_argument("--density", type=float, default=0.5)
    single.add_argument("--warmup", type=int, default=1)
    single.add_argument("--iter", type=int, default=10)

    args = parser.parse_args()
    if args.warmup < 0 or args.iter <= 0:
        parser.error("--warmup must be nonnegative and --iter must be positive")
    if args.mode == "single":
        if min(args.m, args.n, args.k) <= 0:
            parser.error("-m, -n and -k must be positive")
        if args.groupsize != -1 and args.groupsize <= 0:
            parser.error("--groupsize must be -1 or positive")
        if not 0 <= args.density <= 1:
            parser.error("--density must be between 0 and 1")
    else:
        if any(batch <= 0 for batch in args.batchsizes):
            parser.error("--batchsizes must be positive")
        if any(not 0 <= density <= 1 for density in args.densities):
            parser.error("--densities must be between 0 and 1")
        if args.comparison == "structured-spmv" and any(density != 0.5 for density in args.densities):
            parser.error("structured-spmv requires --densities 0.5")
    return args

def main():
    args = parse_args()
    if args.mode == "single":
        res = benchmark_kernel(
            args.kernel, args.m, args.n, args.k, bits=args.bits,
            groupsize=args.groupsize, density=args.density,
            warmup=args.warmup, iter=args.iter,
        )
        print(res)
        return

    models = {name: MODELS[name] for name in args.models}
    options = {"warmup": args.warmup, "iter": args.iter, "profile": args.profile}
    if args.comparison == "spmv":
        benchmark_spmv_kernels(models, args.densities, **options)
    elif args.comparison == "structured-spmv":
        benchmark_structured_spmv_kernels(models, args.densities, **options)
    elif args.comparison == "exl2":
        benchmark_exl2_kernels(models, args.batchsizes, bits=args.bits, **options)
    else:
        functions = {
            "dense": benchmark_dense_kernels,
            "sparse": benchmark_sparse_kernels,
            "quant": benchmark_quant_kernels,
            "marlin": benchmark_marlin_kernels,
        }
        functions[args.comparison](models, args.batchsizes, **options)

if __name__ == "__main__":
    main()
