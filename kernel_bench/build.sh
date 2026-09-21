#!/usr/bin/env bash
set -e

export TORCH_CUDA_ARCH_LIST="8.6"

echo "[1/5] Installing build tools..."
uv pip install setuptools wheel ninja

echo "[2/5] Building Sparse-Marlin..."
uv pip install -e ./Sparse-Marlin --no-build-isolation

echo "[3/5] Installing MACKO..."
uv pip install -e ./MACKO

echo "[4/5] Triggering MACKO CUDA build..."
uv run python -c "import macko_spmv"

echo "[5/5] Building ExLlamaV2..."
uv pip install -e ./exllamav2 --no-deps --no-build-isolation

echo "Build complete."