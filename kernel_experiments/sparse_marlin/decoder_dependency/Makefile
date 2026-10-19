NVCC ?= nvcc
PYTHON ?= python3
.PHONY: all run sass check profile analyze
all: experiment
experiment: experiment.cu decoder.cuh
	@mkdir -p results
	$(NVCC) -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v $< -o $@ 2> results/build.log
run: experiment
	./experiment > results/timing.csv 2> results/run.log
sass: experiment
	cuobjdump --dump-sass experiment > results/sass.txt
check: experiment
	compute-sanitizer --tool memcheck --error-exitcode 1 ./experiment verify
profile: experiment
	$(PYTHON) profile.py
analyze:
	$(PYTHON) analyze.py

.PHONY: pair-run pair-check pair-profile pair-analyze
pair_experiment: pair.cu pair.cuh decoder.cuh
	@mkdir -p results
	$(NVCC) -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v $< -o $@ 2> results/pair_build.log
pair-run: pair_experiment
	./pair_experiment > results/pair_timing.csv 2> results/pair_run.log
	cuobjdump --dump-sass pair_experiment > results/pair_sass.txt
pair-check: pair_experiment
	compute-sanitizer --tool memcheck --error-exitcode 1 ./pair_experiment verify
pair-profile: pair_experiment
	$(PYTHON) pair_profile.py
pair-analyze:
	$(PYTHON) pair_analyze.py
