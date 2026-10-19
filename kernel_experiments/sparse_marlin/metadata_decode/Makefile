ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= $(ROOT)/../../../kernel_bench/.venv/bin/python
MAX_JOBS ?= 2
ARGS ?=
.PHONY: build smoke
build:
	cd "$(ROOT)" && MAX_JOBS=$(MAX_JOBS) "$(PYTHON)" setup.py build_ext --inplace
smoke: build
	"$(PYTHON)" "$(ROOT)/smoke.py" $(ARGS)
