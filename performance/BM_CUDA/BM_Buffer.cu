#include "../performance.h"

using namespace elementwise_SplittableMatrix;

#define MIN_RANGE 1 << 5
#define MAX_RANGE 1 << 14
#define BENCHMARK_APPLY()           \
    RangeMultiplier(2)              \
    ->MeasureProcessCPUTime()       \
    ->UseRealTime()                 \
	->Range(MIN_RANGE, MAX_RANGE)

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
void MAIN_INIT(int argc, char** argv) {
#pragma clang diagnostic pop
    printf(GREEN "==========================\n" RESET);
    OMP_CONSOLE_INFO();
    printf(GREEN "==========================\n" RESET);
    printf("PRECISION_STATUS = %s, MODE = %d\n", PRECISION_STATUS, PRECISION_MODE);
    printf("MATMUL_ORDER = %s\n", MATMUL_ORDER == MATMUL_IJK ? "IJK" : "IKJ");
    benchmark::AddCustomContext("CUDA_ENABLE", CUDA_ENABLE ? "Enable" : "Disable");
    benchmark::AddCustomContext("OMP_ENABLE", OMP_ENABLE ? "Enable" : "Disable");
    benchmark::AddCustomContext("OMP_NUM_THREADS", OMP_ENABLE ? std::to_string(omp_get_max_threads()) : "-1");
	benchmark::AddCustomContext("MUL_RECURSIVE_THRESHOLD", std::to_string(MUL_RECURSIVE_THRESHOLD));
}

BENCHMARK_DEFINE_F(BinaryFixture, BM_MatMul_ForkJoin)(benchmark::State& state) {
	ufunc::matmul::OmpForkJoin<float>::set_threshold(64); // TODO: Fix
	for (auto _ : state) {
		ufunc::matmul::OmpForkJoin<float>::operate(*out, *lhs, *rhs);
	}
}
BENCHMARK_REGISTER_F(BinaryFixture, BM_MatMul_ForkJoin)->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(BinaryFixture, BM_MatMul_CUDA_Tiling)(benchmark::State& state) {
    for (auto _ : state) {
        ufunc::matmul::Tited<float>::operate(*out->root, *lhs->root, *rhs->root);
    }
}
BENCHMARK_REGISTER_F(BinaryFixture, BM_MatMul_CUDA_Tiling)->BENCHMARK_APPLY();

CUSTOM_BENCHMARK_MAIN();
