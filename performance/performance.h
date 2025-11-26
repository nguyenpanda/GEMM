#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include "gemm.h"
#include "fixture.h"
#include <benchmark/benchmark.h>
#include <chrono>

// MPI support (optional - only include if using MPI benchmarks)
#ifdef MPI_ENABLE
#include <mpi.h>
#include "fixtured.h"
#endif

static size_t MATRIX_RDIM;
static size_t MATRIX_CDIM;
static size_t ADD_RECURSIVE_THRESHOLD;
static size_t MUL_RECURSIVE_THRESHOLD;

template<class T>
int INIT_METADATA() {
	printf("OMP_NUM_THREADS = %d\n", omp_get_max_threads());

	ADD_RECURSIVE_THRESHOLD = static_cast<size_t>(atoi(getenv("ADD_RECURSIVE_THRESHOLD")));
	ufunc::addition::OmpForkJoin<T>::set_threshold(ADD_RECURSIVE_THRESHOLD);
	printf("ADD_RECURSIVE_THRESHOLD = %zu\n", ADD_RECURSIVE_THRESHOLD);

	MUL_RECURSIVE_THRESHOLD = static_cast<size_t>(atoi(getenv("MUL_RECURSIVE_THRESHOLD")));
	ufunc::matmul::OmpForkJoin<T>::set_threshold(MUL_RECURSIVE_THRESHOLD);
	printf("MUL_RECURSIVE_THRESHOLD = %zu\n", MUL_RECURSIVE_THRESHOLD);

	printf("PRECISION_STATUS = %s, MODE = %d\n", PRECISION_STATUS, PRECISION_MODE);
	printf("MATMUL_ORDER = %s\n", MATMUL_ORDER == MATMUL_IJK ? "IJK" : "IKJ");

	return 0;
}

void MAIN_INIT(int argc, char** argv);

#define CUSTOM_BENCHMARK_MAIN() 											\
	int main(int argc, char** argv) { 										\
		MAIN_INIT(argc, argv);										 		\
		char arg0_default[] = "benchmark"; 									\
		char* args_default = reinterpret_cast<char*>(arg0_default); 		\
		if (!argv) { 														\
		argc = 1; 															\
		argv = &args_default; 												\
		} 																	\
		benchmark::Initialize(&argc, argv); 								\
		if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1; 	\
		benchmark::RunSpecifiedBenchmarks(); 								\
		benchmark::Shutdown(); 												\
		return 0; 															\
	} 																		\
	int main(int, char**)

#ifdef MPI_ENABLE
// NullReporter disables output from non-root MPI ranks
class NullReporter : public ::benchmark::BenchmarkReporter {
public:
    bool ReportContext(const Context&) override { return true; }
    void ReportRuns(const std::vector<Run>&) override {}
    void Finalize() override {}
};

#define CUSTOM_MPI_BENCHMARK_MAIN()                                         \
    int main(int argc, char** argv) {                                       \
        MPI_Init(&argc, &argv);                                             \
        int rank;                                                           \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                               \
        MPI_Barrier(MPI_COMM_WORLD);                                       \
        /* Filter benchmark_out flags BEFORE Initialize */                  \
        if (rank != 0) {                                                    \
            static std::vector<std::string> args_storage; \
            static std::vector<std::string> filtered_storage; \
            args_storage.assign(argv, argv + argc); \
            filtered_storage.clear();                            \
             for (auto& s : args_storage) {                      \
                if (s.rfind("--benchmark_out", 0) == 0 ||             \
                    s.rfind("--benchmark_out_format", 0) == 0) {      \
                    continue;                                               \
                }                                                           \
                filtered_storage.push_back(s);                               \
            }                                                               \
                                                                            \
            /* Rebuild argc/argv */                                         \
            static std::vector<char*> new_argv;                             \
            new_argv.clear();                                               \
            for (auto& s : filtered_storage) new_argv.push_back(const_cast<char*>(s.c_str())); \
            argc = (int)new_argv.size();                                    \
            argv = new_argv.data();                                         \
        }                                                                   \
                                                                            \
        MAIN_INIT(argc, argv);                                              \
        ::benchmark::Initialize(&argc, argv);                               \
                                                                            \
        if (rank == 0) {                                                    \
            ::benchmark::RunSpecifiedBenchmarks();                          \
        } else {                                                            \
            NullReporter null;                                              \
            ::benchmark::RunSpecifiedBenchmarks(&null);                     \
        }                                                                   \
        MPI_Barrier(MPI_COMM_WORLD);                                        \
        printf("Rank %d: BEFORE Finalize\n", rank);                         \
        fflush(stdout);                                                     \
        MPI_Finalize();                                                     \
        printf("Rank %d: AFTER Finalize\n", rank);                         \
        fflush(stdout);                                                     \
        return 0;                                                           \
    }                                                                       \
    int main(int, char**)

#define CUSTOM_HYBRID_BENCHMARK_MAIN()                                         \
    int main(int argc, char** argv) {                                       \
        int provided;                                                       \
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);      \
        if (provided < MPI_THREAD_MULTIPLE) {                               \
            fprintf(stderr, "Error: The MPI library does not provide required threading level\n"); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                  \
        }                                                                 \
                                                                             \
        int rank;                                                           \
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);                               \
                                                                            \
        /* Filter benchmark_out flags BEFORE Initialize */                  \
        if (rank != 0) {                                                    \
            std::vector<std::string> args(argv, argv + argc);               \
            std::vector<std::string> filtered;                              \
            for (size_t i = 0; i < args.size(); i++) {                      \
                if (args[i].rfind("--benchmark_out", 0) == 0 ||             \
                    args[i].rfind("--benchmark_out_format", 0) == 0) {      \
                    continue;                                               \
                }                                                           \
                filtered.push_back(args[i]);                                \
            }                                                               \
            args = filtered;                                                \
                                                                            \
            /* Rebuild argc/argv */                                         \
            static std::vector<char*> new_argv;                             \
            new_argv.clear();                                               \
            for (auto& s : args) new_argv.push_back(const_cast<char*>(s.c_str())); \
            argc = (int)new_argv.size();                                    \
            argv = new_argv.data();                                         \
        }                                                                   \
                                                                            \
        MAIN_INIT(argc, argv);                                              \
        ::benchmark::Initialize(&argc, argv);                               \
                                                                            \
        if (rank == 0) {                                                    \
            ::benchmark::RunSpecifiedBenchmarks();                          \
        } else {                                                            \
            NullReporter null;                                              \
            ::benchmark::RunSpecifiedBenchmarks(&null);                     \
        }                                                                   \
                                                                            \
        MPI_Finalize();                                                     \
        return 0;                                                           \
    }                                                                       \
    int main(int, char**)
#endif // MPI_ENABLE

#endif // PERFORMANCE_H
