ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BENCH_PYTHON := $(ROOT)/../../kernel_bench/.venv/bin/python
PYTHON ?= $(if $(wildcard $(BENCH_PYTHON)),$(BENCH_PYTHON),python3)
MAX_JOBS ?= 2
ARGS ?=

.PHONY: build test bench upstream-test clean
build:
	cd "$(ROOT)/src" && MAX_JOBS=$(MAX_JOBS) "$(PYTHON)" setup.py build_ext --inplace

test: build
	"$(PYTHON)" "$(ROOT)/run.py" check $(ARGS)

bench: build
	"$(PYTHON)" "$(ROOT)/run.py" bench $(ARGS)

upstream-test: build
	cd "$(ROOT)/src" && "$(PYTHON)" test_2_4.py

clean:
	rm -rf "$(ROOT)/src/build"
	rm -f "$(ROOT)"/src/marlin_cuda*.so
