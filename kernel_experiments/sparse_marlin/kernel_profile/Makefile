PYTHON ?= ../../../kernel_bench/.venv/bin/python
.PHONY: build profile bench check analyze smoke batch-bottleneck
build:
	MAX_JOBS=2 $(PYTHON) setup.py build_ext --inplace > build.log 2>&1
profile: build
	$(PYTHON) profile.py baseline
	$(PYTHON) profile.py ablations
bench: build
	$(PYTHON) bench.py
check: build
	compute-sanitizer --tool memcheck --error-exitcode 9 $(PYTHON) run.py profile --dataset uniform --variant no_gmem
	compute-sanitizer --tool synccheck --error-exitcode 9 $(PYTHON) run.py profile --dataset uniform --variant no_gmem
analyze:
	$(PYTHON) analyze.py
smoke:
	$(PYTHON) smoke.py
batch-bottleneck:
	$(PYTHON) batch_bottleneck.py
	$(PYTHON) batch_counters.py
