NVCC ?= nvcc
RESULTS ?= results
MODES ?= 0 1 2 3 4 5 6 7 8 9 10
.PHONY: all run check sass profile analyze
all: experiment
experiment: experiment.cu
	@mkdir -p results
	$(NVCC) -O3 -std=c++17 -arch=sm_86 -lineinfo -Xptxas=-v $< -o $@ 2> results/build.log
run: experiment
	@mkdir -p $(RESULTS)
	./experiment > $(RESULTS)/timing.csv 2> $(RESULTS)/run.log
check: experiment
	compute-sanitizer --tool memcheck --error-exitcode 1 ./experiment verify
	compute-sanitizer --tool synccheck --error-exitcode 1 ./experiment verify
sass: experiment
	@mkdir -p $(RESULTS)
	cuobjdump --dump-sass experiment > $(RESULTS)/sass.txt
profile: experiment
	@mkdir -p $(RESULTS)
	@for mode in $(MODES); do \
		ncu --clock-control none --cache-control all --kernel-name-base function \
		--kernel-name metadata_load --launch-count 1 --set detailed -f \
		-o $(RESULTS)/mode$$mode ./experiment profile $$mode \
		> $(RESULTS)/profile_mode$$mode.log 2>&1 || exit 1; \
		ncu --import $(RESULTS)/mode$$mode.ncu-rep --csv --page raw \
		> $(RESULTS)/raw_mode$$mode.csv || exit 1; \
	done
analyze:
	python3 analyze.py $(RESULTS)
