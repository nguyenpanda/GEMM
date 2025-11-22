#include "../performance.h"
#include <set>
#include <string>

using namespace elementwise_SplittableMatrix_Distributed;

#define MIN_RANGE 1 << 2
#define MAX_RANGE 1 << 14
#define BENCHMARK_APPLY()           \
    RangeMultiplier(2)              \
    ->UseManualTime()               \
    ->Range(MIN_RANGE, MAX_RANGE)

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
void MAIN_INIT(int argc, char** argv) {
#pragma clang diagnostic pop
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get processor/node name for each rank
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    // Gather all processor names to rank 0
    std::vector<char> all_names;
    if (rank == 0) {
        all_names.resize(size * MPI_MAX_PROCESSOR_NAME);
    }
    
    MPI_Gather(processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               all_names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf(GREEN "==========================\n" RESET);
        printf("MPI Baseline Matrix Multiplication (IKJ)\n");
        printf(GREEN "==========================\n" RESET);
        
        printf("MPI_COMM_SIZE = %d\n", size);
        printf("Strategy: Row-wise distribution with IKJ loop order\n");
        
        // Build hostname list
        std::string hosts_str;
        std::set<std::string> unique_hosts;
        for (int i = 0; i < size; ++i) {
            std::string host(&all_names[i * MPI_MAX_PROCESSOR_NAME]);
            unique_hosts.insert(host);
            if (i > 0) hosts_str += ", ";
            hosts_str += "rank" + std::to_string(i) + "=" + host;
        }
        
        printf("Hosts: %s\n", hosts_str.c_str());
        printf("Unique nodes: \n");
        for (const auto& host : unique_hosts) {
            printf(" - %s\n", host.c_str());
        }
        
        benchmark::AddCustomContext("MPI_COMM_SIZE", std::to_string(size));
        benchmark::AddCustomContext("strategy", "Row-wise IKJ Async");
        benchmark::AddCustomContext("processes", hosts_str);
        // Add count and comma-separated list of unique hostnames to benchmark context
        benchmark::AddCustomContext("node_number", std::to_string(unique_hosts.size()));
        std::string unique_hosts_str;
        for (auto it = unique_hosts.begin(); it != unique_hosts.end(); ++it) {
            if (it != unique_hosts.begin()) unique_hosts_str += ", ";
            unique_hosts_str += *it;
        }
        benchmark::AddCustomContext("node_names", unique_hosts_str);
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
            ufunc::matmuld::MPIBaseline<float>::operate(*out, *lhs, *rhs);                    \
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
    BENCHMARK_REGISTER_F(BinaryFixture, BM_MPI_Baseline)->BENCHMARK_APPLY()

BENCHMARK_MPI_BASELINE();

CUSTOM_MPI_BENCHMARK_MAIN();
