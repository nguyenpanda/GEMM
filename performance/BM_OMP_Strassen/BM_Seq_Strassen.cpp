#include "../performance.h"

using namespace elementwise_SplittableMatrix;

#define MIN_RANGE 1 << 5
#define MAX_RANGE 1 << 5
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
    benchmark::AddCustomContext("OMP_ENABLE", OMP_ENABLE ? "Enable" : "Disable");
    benchmark::AddCustomContext("OMP_NUM_THREADS", OMP_ENABLE ? std::to_string(omp_get_max_threads()) : "-1");
}

#define THRESHOLD_LIST \
    X(2)               \
    X(4)               \
    X(8)               \
    X(16)              \
    X(32)              \
    X(64)              \
    X(128)             \
    X(256)             \
    X(512)             \
    X(1024)            \
    X(2048)            \


#define BENCHMARK_FORKJOIN_THRESHOLD(THRESHOLD)                             \
    BENCHMARK_DEFINE_F(BinaryFixture, BM_Seq_Strassen_T##THRESHOLD)         \
    (benchmark::State& state) {                                             \
        ufunc::matmul::SeqStrassen<float>::set_threshold(THRESHOLD);       \
        for (auto _ : state) {                                              \
            ufunc::matmul::SeqStrassen<float>::operate(*out, *lhs, *rhs);                                                                  \
    }                                                                       \
    }                                                                       \
    BENCHMARK_REGISTER_F(BinaryFixture, BM_Seq_Strassen_T##THRESHOLD)->BENCHMARK_APPLY();

#define X(v) BENCHMARK_FORKJOIN_THRESHOLD(v);
    THRESHOLD_LIST
#undef X
    
CUSTOM_BENCHMARK_MAIN();

// 
