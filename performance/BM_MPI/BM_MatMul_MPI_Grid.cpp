#include "../performance.h"

using namespace elementwise_SplittableMatrix_Distributed;

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

#define BENCHMARK_MPI_GRID_MATMUL()                                                          \
    BENCHMARK_DEFINE_F(BinaryFixture, BM_MPI_GridMatMul)                                     \
    (benchmark::State& state) {                                                              \
        double max_elapsed_second;                                                           \
        int rank;                                                                            \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                                \
        for (auto _ : state) {                                                               \
            MPI_Barrier(MPI_COMM_WORLD);                                                     \
            auto start = std::chrono::high_resolution_clock::now();                          \
            ufunc::matmuld::MPIGridForkJoin<float>::operate(*out, *lhs, *rhs);                \
            auto end = std::chrono::high_resolution_clock::now();                            \
            MPI_Barrier(MPI_COMM_WORLD);                                                     \
            auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(       \
                end - start);                                                                \
            double elapsed_seconds = duration.count();                                       \
            MPI_Allreduce(&elapsed_seconds, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX,    \
                          MPI_COMM_WORLD);                                                   \
            state.SetIterationTime(max_elapsed_second);                                      \
        }                                                                                    \
    }                                                                                        \
    BENCHMARK_REGISTER_F(BinaryFixture, BM_MPI_GridMatMul)->BENCHMARK_APPLY();

BENCHMARK_MPI_GRID_MATMUL()

CUSTOM_MPI_BENCHMARK_MAIN();