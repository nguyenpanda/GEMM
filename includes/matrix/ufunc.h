#ifndef MATRIX_UFUNC_H
#define MATRIX_UFUNC_H

#include "ufunc/add.h"
#include "ufunc/matmul.h"

#if CUDA_ENABLE == 1
#include "ufunc/matmul.cuh"
#endif

#ifdef MPI_ENABLE
#include "ufunc/matmuld.h"
#endif

#endif // MATRIX_UFUNC_H
