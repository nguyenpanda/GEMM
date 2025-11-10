#include "../performance.h"

using namespace elementwise_SplittableMatrix;

#define MIN_RANGE 1 << 6
#define MAX_RANGE 1 << 12
#define BENCHMARK_APPLY()           \
    RangeMultiplier(2)              \
    ->UseManualTime()               \
    ->Unit(benchmark::kSecond)      \
    ->Range(MIN_RANGE, MAX_RANGE)

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
void MAIN_INIT(int argc, char** argv) {
#pragma clang diagnostic pop
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        printf(GREEN "==========================\n" RESET);
        printf("MPI Baseline Matrix Multiplication (IKJ)\n");
        printf(GREEN "==========================\n" RESET);
        
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        printf("MPI_COMM_SIZE = %d\n", size);
        printf("Strategy: Row-wise distribution with IKJ loop order\n");
        
        benchmark::AddCustomContext("MPI_COMM_SIZE", std::to_string(size));
        benchmark::AddCustomContext("Strategy", "Row-wise IKJ");
    }
}

#define BENCHMARK_MPI_BASELINE()                                                             \
    BENCHMARK_DEFINE_F(BinaryFixture, BM_MPI_Baseline)                                       \
    (benchmark::State& state) {                                                              \
        double max_elapsed_second;                                                           \
        int rank, size;                                                                      \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                                \
        MPI_Comm_size(MPI_COMM_WORLD, &size);                                                \
                                                                                             \
        if (size < 2) {                                                                      \
            if (rank == 0) {                                                                 \
                state.SkipWithError("Need at least 2 processes (1 master + 1 worker)");      \
            }                                                                                \
            return;                                                                          \
        }                                                                                    \
                                                                                             \
        for (auto _ : state) {                                                               \
            MPI_Barrier(MPI_COMM_WORLD);                                                     \
            auto start = std::chrono::high_resolution_clock::now();                          \
                                                                                             \
            /* Use MPIBaseline class from matmul.h */                                       \
            ufunc::matmul::MPIBaseline<float>::operate(*out, *lhs, *rhs);                    \
                                                                                             \
            MPI_Barrier(MPI_COMM_WORLD);                                                     \
            auto end = std::chrono::high_resolution_clock::now();                            \
                                                                                             \
            auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(       \
                end - start);                                                                \
            double elapsed_seconds = duration.count();                                       \
            MPI_Allreduce(&elapsed_seconds, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX,    \
                          MPI_COMM_WORLD);                                                   \
            state.SetIterationTime(max_elapsed_second);                                      \
        }                                                                                    \
    }                                                                                        \
    BENCHMARK_REGISTER_F(BinaryFixture, BM_MPI_Baseline)->BENCHMARK_APPLY();

BENCHMARK_MPI_BASELINE()

CUSTOM_MPI_BENCHMARK_MAIN();
