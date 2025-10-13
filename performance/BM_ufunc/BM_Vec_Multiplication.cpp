#include "../performance.h"

using namespace elementwise_Buffer;

#define MIN_RANGE 1 << 1
#define MAX_RANGE 1 << 16
#define BENCHMARK_APPLY()           \
    RangeMultiplier(2)              \
    ->MeasureProcessCPUTime()       \
    ->UseRealTime()                 \
	->Range(MIN_RANGE, MAX_RANGE)

void MAIN_INIT(int argc, char** argv) {
    printf(GREEN "==========================\n" RESET);
    OMP_CONSOLE_INFO();
    printf(GREEN "==========================\n" RESET);
    benchmark::AddCustomContext("OpenMP status", OMP_ENABLE ? "Enable" : "Disable");
    benchmark::AddCustomContext("OMP_NUM_THREADS", OMP_ENABLE ? std::to_string(omp_get_max_threads()) : "-1");
}

BENCHMARK_DEFINE_F(BinaryFixture, BM_Vec_Multiplication_Seq)(benchmark::State& state) {
    for (auto _ : state) {
        for (size_t i = 0; i < out->size(); i++) {
            out->data[i] = lhs->data[i] * rhs->data[i];
        }
    }
}

BENCHMARK_REGISTER_F(BinaryFixture, BM_Vec_Multiplication_Seq)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(BinaryFixture, BM_Vec_Multiplication_Omp_Vanilla)(benchmark::State& state) {
    for (auto _ : state) {
        #pragma omp parallel for
        for (size_t i = 0; i < out->size(); i++) {
            out->data[i] = lhs->data[i] * rhs->data[i];
        }
    }
}

BENCHMARK_REGISTER_F(BinaryFixture, BM_Vec_Multiplication_Omp_Vanilla)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(BinaryFixture, BM_Vec_Multiplication_Omp_ForSIMD)(benchmark::State& state) {
    for (auto _ : state) {
        #pragma omp parallel
        {
            #pragma omp for simd
            for (size_t i = 0; i < out->size(); i++) {
                out->data[i] = lhs->data[i] * rhs->data[i];
            }
        }
        
    }
}

BENCHMARK_REGISTER_F(BinaryFixture, BM_Vec_Multiplication_Omp_ForSIMD)
	->BENCHMARK_APPLY();

CUSTOM_BENCHMARK_MAIN();
