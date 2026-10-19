NVCC ?= nvcc
FLAGS = -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v
METRICS = l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,dram__sectors_read.sum,dram__sectors_write.sum,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__cycles_elapsed.avg,gpu__time_duration.sum
.PHONY: all run profile check analyze clean
all: experiment results/sass.txt
experiment: experiment.cu
	@mkdir -p results
	$(NVCC) $(FLAGS) $< -o $@ 2> results/build.log
results/sass.txt: experiment
	cuobjdump --dump-sass experiment > $@
run: all
	./experiment > results/measurements.csv 2> results/run.log
profile: all
	@for m in 0 1 2 3 4 5 6 7 8 9; do ncu --clock-control none --cache-control all --kernel-name-base function --kernel-name expansion --launch-skip 1 --launch-count 1 --metrics $(METRICS) --csv ./experiment profile $$m > results/ncu_$$m.csv 2> results/ncu_$$m.err || exit 1; done
check: all
	compute-sanitizer --tool memcheck --error-exitcode 9 ./experiment verify > results/memcheck.log 2>&1
	compute-sanitizer --tool synccheck --error-exitcode 9 ./experiment verify > results/synccheck.log 2>&1
analyze:
	python3 analyze.py
	python3 report.py
clean:
	rm -f experiment
