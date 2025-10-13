#ifndef OMP_INTRO_H
#define OMP_INTRO_H

#include "matrix.h"
#include "utils.h"

#include <iostream>
#include <chrono>
#include <cmath>

#ifdef _OPENMP
	#define OMP_CONSOLE_INFO() \
		printf(YELLOW "OMP_NUM_THREADS = %d\n" RESET, omp_get_max_threads())
#else
	#define OMP_CONSOLE_INFO() \
		printf(YELLOW "DISABLE OpenMP!\n" RESET)
#endif // _OPENMP

#endif // OMP_INTRO_H
