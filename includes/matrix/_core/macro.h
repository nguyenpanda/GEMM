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
#else
    #ifdef _OPENMP
        #include <omp.h>
    #else
        #error "OMP_ENABLE but _OPENMP is not defined. Add OpenMP flags to your compiler (e.g., -fopenmp)."
    #endif
#endif // OMP_ENABLE == 0

// OPENMPI
#ifdef MPI_ENABLE
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#endif

// CUDA
#ifdef __NVCC__
#define CUDA_ENABLE 1
#include <cuda_runtime.h>

#define CUDA_CHECK(err) { 																\
	if (err != cudaSuccess) { 															\
		printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__); 	\
		exit(EXIT_FAILURE);																\
	}																					\
}
#else
#define CUDA_ENABLE 0
#endif // #ifndef CUDA_ENABLE

// DEBUG MODE
#ifndef DEBUG_MODE
	#define CODE_FOR_DEBUG_MODE(...)
	#define COMPILER_MODE "RELEASE"
#else
	#define CODE_FOR_DEBUG_MODE(...) __VA_ARGS__
	#define COMPILER_MODE "DEBUG"
#endif // ifndef DEBUG_MODE

////////////////////////////////////
////////    MATRIX SETUP    ////////
////////////////////////////////////

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

// MATRIX MULTIPLICATION ON CUDA
#if CUDA_ENABLE == 1
#ifndef CUDA_TILE_WIDTH
#define CUDA_TILE_WIDTH 64
#endif // #ifndef CUDA_TILE_WIDTH
#endif // CUDA_ENABLE == 1

#endif // _MATRIX__CORE_MACRO_H
