#ifndef _MATRIX__CORE_MACRO_H
#define _MATRIX__CORE_MACRO_H

// OPENMP
#ifndef OMP_ENABLE
    #define OMP_ENABLE 0
#endif // #ifndef OMP_ENABLE

#if OMP_ENABLE == 0
    #undef _OPENMP
    #define omp_get_max_threads() 1
    #define omp_get_num_procs() 1
	#define omp_get_thread_num() 0
	void HATUONGNGUYEN_OPENMP_ZERO() {
		return;
	}
#else
    #ifdef _OPENMP
        #include <omp.h>
    #else
        #error "OMP_ENABLE but _OPENMP is not defined. Add OpenMP flags to your compiler (e.g., -fopenmp)."
    #endif
#endif // OMP_ENABLE == 0

// DEBUG MODE
#ifndef DEBUG_MODE
	#define CODE_FOR_DEBUG_MODE(...)
	#define COMPILER_MODE "RELEASE"
#else
	#define CODE_FOR_DEBUG_MODE(...) __VA_ARGS__
	#define COMPILER_MODE "DEBUG"
#endif // ifndef DEBUG_MODE

// MATRIX MULTIPLICATION ORDER
#define MATMUL_IJK 1
#define MATMUL_IKJ 2

#if MATMUL_ORDER == 0 || !defined(MATMUL_ORDER)
	#undef MATMUL_ORDER
	#define MATMUL_ORDER MATMUL_IJK
#endif // MATMUL_ORDER == 0 || !defined(MATMUL_ORDER)

// PRECISION MODE
#define PRECISION_KANAN 1 // William Kahan Summation Algorithm
#define PRECISION_NEUMAIER 2 // Improved Kahan–Babuška Algorithm

#if PRECISION_MODE == 0 || !defined(PRECISION_MODE)
	#undef PRECISION_MODE
	#define PRECISION_MODE 0
    #define PRECISION_STATUS "OFF"
#else
    #define PRECISION_STATUS "ON"
#endif // PRECISION_MODE == 0 || !defined(PRECISION_MODE)

#endif // _MATRIX__CORE_MACRO_H
