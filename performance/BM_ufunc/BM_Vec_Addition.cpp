#include "../performance.h"
#include <immintrin.h>

typedef __m256 simd256;

using namespace elementwise_Buffer;

#define MIN_RANGE 1 << 2
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

BENCHMARK_DEFINE_F(BinaryFixture, BM_Vec_Addition_Seq)(benchmark::State& state) {
    for (auto _ : state) {
        for (size_t i = 0; i < out->size(); i++) {
            out->data[i] = lhs->data[i] + rhs->data[i];
        }
    }
}

BENCHMARK_REGISTER_F(BinaryFixture, BM_Vec_Addition_Seq)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(BinaryFixture, BM_Vec_Addition_Omp_Vanilla)(benchmark::State& state) {
    for (auto _ : state) {
        #pragma omp parallel for
        for (size_t i = 0; i < out->size(); i++) {
            out->data[i] = lhs->data[i] + rhs->data[i];
        }
    }
}

BENCHMARK_REGISTER_F(BinaryFixture, BM_Vec_Addition_Omp_Vanilla)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(BinaryFixture, BM_Vec_Addition_Omp_SIMD)(benchmark::State& state) {
    for (auto _ : state) {
        #pragma omp simd
        for (size_t i = 0; i < out->size(); i++) {
            out->data[i] = lhs->data[i] + rhs->data[i];
        }
    }
}

BENCHMARK_REGISTER_F(BinaryFixture, BM_Vec_Addition_Omp_SIMD)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(BinaryFixture, BM_Vec_Addition_Omp_ParForSIMD)(benchmark::State& state) {
    for (auto _ : state) {
        #pragma omp parallel
        {
            #pragma omp for simd
            for (size_t i = 0; i < out->size(); i++) {
                out->data[i] = lhs->data[i] + rhs->data[i];
            }
        }
        
    }
}

BENCHMARK_REGISTER_F(BinaryFixture, BM_Vec_Addition_Omp_ParForSIMD)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(BinaryFixture, BM_Vec_Addition_Omp_ParFor_Immintrin)(benchmark::State& state) {
    for (auto _ : state) {
        const size_t N = out->size();

        #pragma omp parallel for
        for (size_t i = 0; i < N; i += 8) {
            simd256 simd256_lhs = _mm256_load_ps(lhs->data + i);
            simd256 simd256_rhs = _mm256_load_ps(rhs->data + i);
            simd256 simd256_out = _mm256_add_ps(simd256_lhs, simd256_rhs);
            _mm256_store_ps(out->data + i, simd256_out);
        }

        const size_t start_r = static_cast<size_t>(N / 8) * 8;
        for (size_t i = start_r; i < N; i++) {
            out->data[i] = lhs->data[i] + rhs->data[i];
        }
    }
}

BENCHMARK_REGISTER_F(BinaryFixture, BM_Vec_Addition_Omp_ParFor_Immintrin)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(BinaryFixture, BM_Vec_Addition_Immintrin)(benchmark::State& state) {
    for (auto _ : state) {
        const size_t N = out->size();

        for (size_t i = 0; i < N; i += 8) {
            simd256 simd256_lhs = _mm256_load_ps(lhs->data + i);
            simd256 simd256_rhs = _mm256_load_ps(rhs->data + i);
            simd256 simd256_out = _mm256_add_ps(simd256_lhs, simd256_rhs);
            _mm256_store_ps(out->data + i, simd256_out);
        }

        const size_t start_r = static_cast<size_t>(N / 8) * 8;
        for (size_t i = start_r; i < N; i++) {
            out->data[i] = lhs->data[i] + rhs->data[i];
        }
    }
}

BENCHMARK_REGISTER_F(BinaryFixture, BM_Vec_Addition_Immintrin )
	->BENCHMARK_APPLY();

CUSTOM_BENCHMARK_MAIN();
