#include "../performance.h"

using namespace elementwise_SplittableMatrix;

#define MIN_RANGE 1 << 1
#define MAX_RANGE 1 << 16
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

BENCHMARK_DEFINE_F(BinaryFixture, BM_MatMul_Seq_ijk)(benchmark::State& state) {
    for (auto _ : state) {
        for (size_t i = 0; i < out->rdim; i++) {
            for (size_t j = 0; j < out->cdim; j++) {
                float c = 0.0f;
                for (size_t k = 0; k < lhs->cdim; k++) {
                    const size_t lhs_idx = lhs->map2Dto1DIndex(i, k);
                    const size_t rhs_idx = rhs->map2Dto1DIndex(k, j);
                    const float lhs_val = lhs->get(lhs_idx);
                    const float rhs_val = rhs->get(rhs_idx);
                    c += lhs_val * rhs_val;
                }
                const size_t out_idx = out->map2Dto1DIndex(i, j);
                out->set(out_idx, c);
            }
        }
    }
}

BENCHMARK_REGISTER_F(BinaryFixture, BM_MatMul_Seq_ijk)
	->BENCHMARK_APPLY();

BENCHMARK_DEFINE_F(BinaryFixture, BM_MatMul_Seq_ikj)(benchmark::State& state) {
    for (auto _ : state) {
        for (size_t i = 0; i < out->rdim; i++) {
            for (size_t k = 0; k < lhs->cdim; k++) {
                const size_t lhs_idx = lhs->map2Dto1DIndex(i, k);
                const float lhs_val = lhs->get(lhs_idx);
                for (size_t j = 0; j < out->cdim; j++) {
                    const size_t rhs_idx = rhs->map2Dto1DIndex(k, j);
                    const size_t out_idx = out->map2Dto1DIndex(i, j);
                    const float rhs_val = rhs->get(rhs_idx);
                    const float prev_out = out->get(out_idx);
                    
                    out->set(out_idx, prev_out + lhs_val * rhs_val);
                }
            }
        }
    }
}

BENCHMARK_REGISTER_F(BinaryFixture, BM_MatMul_Seq_ikj)
	->BENCHMARK_APPLY();

// BENCHMARK_DEFINE_F(BinaryFixture, BM_MatMul_Omp_ijk_Vanilla)(benchmark::State& state) {
//     // TODO
// }

// BENCHMARK_REGISTER_F(BinaryFixture, BM_MatMul_Omp_ijk_Vanilla)
// 	->BENCHMARK_APPLY();

// BENCHMARK_DEFINE_F(BinaryFixture, BM_MatMul_Omp_ikj_Vanilla)(benchmark::State& state) {
//     // TODO
// }

// BENCHMARK_REGISTER_F(BinaryFixture, BM_MatMul_Omp_ikj_Vanilla)
// 	->BENCHMARK_APPLY();

// BENCHMARK_DEFINE_F(BinaryFixture, BM_MatMul_Omp_ikj_Vanilla)(benchmark::State& state) {
//     // TODO
// }

// BENCHMARK_REGISTER_F(BinaryFixture, BM_MatMul_Omp_ikj_Vanilla)
// 	->BENCHMARK_APPLY();

CUSTOM_BENCHMARK_MAIN();
