#include "../performance.h"

using namespace elementwise_SplittableMatrix;

#define MIN_RANGE 1 << 2
#define MAX_RANGE 1 << 13
#define BENCHMARK_APPLY()           \
    RangeMultiplier(2)              \
    ->MeasureProcessCPUTime()       \
    ->UseRealTime()                 \
	->Range(MIN_RANGE, MAX_RANGE) \
    ->Iterations(1) 

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
void MAIN_INIT(int argc, char** argv) {
#pragma clang diagnostic pop
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        printf(GREEN "==========================\n" RESET);
        printf("MPI Grid Matrix Multiplication\n");
        printf(GREEN "==========================\n" RESET);
        
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        printf("MPI_COMM_SIZE = %d\n", size);
        printf("GRID_SIZE = %d x %d\n", (int)std::sqrt(size), (int)std::sqrt(size));
        
        benchmark::AddCustomContext("MPI_COMM_SIZE", std::to_string(size));
        benchmark::AddCustomContext("GRID_SIZE", std::to_string((int)std::sqrt(size)));
    }
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
    
#define BENCHMARK_MPI_STRASSEN()                                              \
    BENCHMARK_DEFINE_F(BinaryFixture, BM_MPI_Strassen)                           \
    (benchmark::State& state) {                                                  \
        double max_elapsed = 0.0;                                            \
        int rank;                                                                \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                    \
        ufunc::matmul::MPIStrassen<float>::set_threshold(32);                                                         \
        for (auto _ : state) {                                                    \
            MPI_Barrier(MPI_COMM_WORLD);                                         \
                                                                               \
            auto t0 = std::chrono::steady_clock::now();                          \
                                                                                 \
            ufunc::matmul::MPIStrassen<float>::operate(*out, *lhs, *rhs);        \
            MPI_Barrier(MPI_COMM_WORLD);                                         \
            auto t1 = std::chrono::steady_clock::now();                          \
            double local_elapsed =                                               \
                std::chrono::duration<double>(t1 - t0).count();                  \
            MPI_Reduce(&local_elapsed, &max_elapsed, 1,                          \
                       MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);                  \
                                                                                 \
            state.SetIterationTime(max_elapsed);                                 \
        }                                                                        \
                MPI_Barrier(MPI_COMM_WORLD);                                       \
    }                                                                            \
    BENCHMARK_REGISTER_F(BinaryFixture, BM_MPI_Strassen)->BENCHMARK_APPLY();

// #define X(v) BENCHMARK_MPI_STRASSEN(v);
//     THRESHOLD_LIST
// #undef X

BENCHMARK_MPI_STRASSEN();


CUSTOM_MPI_BENCHMARK_MAIN();