#include "../performance.h"

using namespace elementwise_SplittableMatrix;

#define MIN_RANGE 1 << 2
#define MAX_RANGE 1 << 13
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
    INIT_METADATA<float>();
    benchmark::AddCustomContext("OMP_ENABLE", OMP_ENABLE ? "Enable" : "Disable");
    benchmark::AddCustomContext("OMP_NUM_THREADS", OMP_ENABLE ? std::to_string(omp_get_max_threads()) : "-1");
    benchmark::AddCustomContext("ADD_RECURSIVE_THRESHOLD", std::to_string(ADD_RECURSIVE_THRESHOLD));
    benchmark::AddCustomContext("MUL_RECURSIVE_THRESHOLD", std::to_string(MUL_RECURSIVE_THRESHOLD));
    benchmark::AddCustomContext("PRECISION_STATUS", PRECISION_STATUS);
    benchmark::AddCustomContext("PRECISION_MODE", std::to_string(PRECISION_MODE));
    benchmark::AddCustomContext("MATMUL_ORDER", MATMUL_ORDER == MATMUL_IJK ? "IJK" : "IKJ");
}

BENCHMARK_DEFINE_F(BinaryFixture, BM_MatMul_Seq)(benchmark::State& state) {
    for (auto _ : state) {
        ufunc::matmul::Seq<float>::operate(*out, *lhs, *rhs);
    }
}

BENCHMARK_REGISTER_F(BinaryFixture, BM_MatMul_Seq)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(BinaryFixture, BM_MatMul_Omp_Vanilla)(benchmark::State& state) {
    for (auto _ : state) {
		ufunc::matmul::OmpVanilla<float>::operate(*out, *lhs, *rhs);
	}
}

BENCHMARK_REGISTER_F(BinaryFixture, BM_MatMul_Omp_Vanilla)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(BinaryFixture, BM_MatMul_Omp_ForkJoin)(benchmark::State& state) {
    for (auto _ : state) {
		ufunc::matmul::OmpForkJoin<float>::operate(*out, *lhs, *rhs);
	}
}

BENCHMARK_REGISTER_F(BinaryFixture, BM_MatMul_Omp_ForkJoin)
	->BENCHMARK_APPLY();

CUSTOM_BENCHMARK_MAIN();
