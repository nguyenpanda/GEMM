#include "../performance.h"
#include <random>

using namespace elementwise_Buffer;

static thread_local std::mt19937 generator;
static std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

inline float FloatRandom() {
    return distribution(generator);
}

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
    benchmark::AddCustomContext("OMP_ENABLE", OMP_ENABLE ? "Enable" : "Disable");
    benchmark::AddCustomContext("OMP_NUM_THREADS", OMP_ENABLE ? std::to_string(omp_get_max_threads()) : "-1");
}

BENCHMARK_DEFINE_F(UnaryBufferFixture, BM_InitializeRandom_Seq)(benchmark::State& state) {
    for (auto _ : state) {
        for (size_t i = 0; i < buffer->size(); i++) {
            buffer->data[i] = FloatRandom();
        }
    }
}

BENCHMARK_REGISTER_F(UnaryBufferFixture, BM_InitializeRandom_Seq)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(UnaryBufferFixture, BM_InitializeRandom_Omp_Vanilla)(benchmark::State& state) {
    InitializeRandom<float> initialize_random(-1.0f, 1.0f);

    for (auto _ : state) {
        initialize_random.fill(*buffer);
    }
}

BENCHMARK_REGISTER_F(UnaryBufferFixture, BM_InitializeRandom_Omp_Vanilla)
	->BENCHMARK_APPLY();

CUSTOM_BENCHMARK_MAIN();
