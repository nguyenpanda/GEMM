#include "../performance.h"

using namespace elementwise_Buffer;

static thread_local std::mt19937 generator;
static std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

inline float FloatRandom() {
    return distribution(generator);
}

#define MIN_RANGE 1 << 1
#define MAX_RANGE 1 << 17
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

BENCHMARK_DEFINE_F(UnaryBufferFixture, BM_Random_Seq)(benchmark::State& state) {
    for (auto _ : state) {
        for (size_t i = 0; i < buffer->size(); i++) {
            buffer->data[i] = FloatRandom();
        }
    }
}

BENCHMARK_REGISTER_F(UnaryBufferFixture, BM_Random_Seq)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(UnaryBufferFixture, BM_Random_Omp_Vanilla)(benchmark::State& state) {
    for (auto _ : state) {
        #pragma omp parallel for
        for (size_t i = 0; i < buffer->size(); i++) {
            buffer->data[i] = FloatRandom();
        }
    }
}

BENCHMARK_REGISTER_F(UnaryBufferFixture, BM_Random_Omp_Vanilla)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(UnaryBufferFixture, BM_Random_Omp_ForSIMD)(benchmark::State& state) {
    for (auto _ : state) {
        #pragma omp parallel for simd
        for (size_t i = 0; i < buffer->size(); i++) {
            buffer->data[i] = FloatRandom();
        }
    }   
}

BENCHMARK_REGISTER_F(UnaryBufferFixture, BM_Random_Omp_ForSIMD)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(UnaryBufferFixture, BM_Random_Omp_IBM)(benchmark::State& state) {
    /*
        Link: https://www.ibm.com/docs/en/xl-c-and-cpp-linux/16.1.0?topic=parallelization-pragma-omp-simd
    */

    for (auto _ : state) {
        const size_t N = buffer->size();
        #pragma omp target map(to: N) map(tofrom: buffer->data)
        #pragma omp parallel for simd
        for (size_t i = 0; i < N; i++) {
            buffer->data[i] = FloatRandom();
        }
    }   
}

BENCHMARK_REGISTER_F(UnaryBufferFixture, BM_Random_Omp_IBM)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(UnaryBufferFixture, BM_Random_Omp_Panda)(benchmark::State& state) {
    size_t rows   = buffer->rdim;
    size_t cols   = buffer->cdim;
    auto* data    = buffer->data;

    constexpr size_t Ti = 64;
    constexpr size_t Tj = 64;

    for (auto _ : state) {
        #pragma omp parallel for collapse(2) schedule(static) firstprivate(rows, cols, data, Ti, Tj)
        for (size_t ii = 0; ii < rows; ii += Ti) {
            for (size_t jj = 0; jj < cols; jj += Tj) {
                size_t i_max = std::min(ii + Ti, rows);
                size_t j_max = std::min(jj + Tj, cols);

                for (size_t i = ii; i < i_max; i++) {
                    #pragma omp simd
                    for (size_t j = jj; j < j_max; j++) {
                        data[i * cols + j]  = FloatRandom();
                    }
                }
            }
        }
    }
}

BENCHMARK_REGISTER_F(UnaryBufferFixture, BM_Random_Omp_Panda)
	->BENCHMARK_APPLY();

CUSTOM_BENCHMARK_MAIN();
