#include "../performance.h"

using namespace elementwise_SplittableMatrix;

#define MIN_RANGE 1 << 2
#define MAX_RANGE 1 << 15
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
    benchmark::AddCustomContext("OMP_ENABLE", OMP_ENABLE ? "Enable" : "Disable");
    benchmark::AddCustomContext("OMP_NUM_THREADS", OMP_ENABLE ? std::to_string(omp_get_max_threads()) : "-1");
}

BENCHMARK_DEFINE_F(BinaryFixture, BM_MatAdd_Seq)(benchmark::State& state) {
    for (auto _ : state) {
        ufunc::addition::Seq<float>::operate(*out, *lhs, *rhs);
    }
}
BENCHMARK_REGISTER_F(BinaryFixture, BM_MatAdd_Seq)->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(BinaryFixture, BM_MatAdd_Vanilla)(benchmark::State& state) {
    for (auto _ : state) {
        ufunc::addition::OmpVanilla<float>::operate(*out, *lhs, *rhs);
    }
}
BENCHMARK_REGISTER_F(BinaryFixture, BM_MatAdd_Vanilla)->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(BinaryFixture, BM_MatAdd_ForkJoin_Optimal)(benchmark::State& state) {
    size_t N = static_cast<size_t>(state.range(0));
    size_t THRESHOLD = static_cast<size_t>(N / 2);
    if (N <= 128) {
        THRESHOLD = 256;
    }
    ufunc::addition::OmpForkJoin<float>::set_threshold(THRESHOLD);

    for (auto _ : state) {
        ufunc::addition::OmpForkJoin<float>::operate(*out, *lhs, *rhs);
    }
}
BENCHMARK_REGISTER_F(BinaryFixture, BM_MatAdd_ForkJoin_Optimal)->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(BinaryFixture, BM_MatAdd_GODDAMNNNN)(benchmark::State& state) {
    for (auto _ : state) {
        ufunc::addition::GODDAMNNN<float>::operate(*out, *lhs, *rhs);
    }
}
BENCHMARK_REGISTER_F(BinaryFixture, BM_MatAdd_GODDAMNNNN)->BENCHMARK_APPLY();
    
CUSTOM_BENCHMARK_MAIN();
