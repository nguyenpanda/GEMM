#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include "gemm.h"
#include "fixture.h"
#include <benchmark/benchmark.h>

void MAIN_INIT(int argc, char** argv);

#define CUSTOM_BENCHMARK_MAIN() 											\
	int main(int argc, char** argv) { 										\
		MAIN_INIT(argc, argv);										 		\
		benchmark::MaybeReenterWithoutASLR(argc, argv); 					\
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

#endif // PERFORMANCE_H
