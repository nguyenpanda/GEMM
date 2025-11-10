#ifndef MATRIX_MPI_H
#define MATRIX_MPI_H

// MPI matrix operations are integrated into ufunc/matmul.h
// Include ufunc header which contains MPIGridForkJoin when USE_MPI is defined
#include "ufunc/matmul.h"

// Alias for backward compatibility
namespace mpi {
namespace matmul {
    using ufunc::matmul::MPIGridForkJoin;
}
}

#endif // MATRIX_MPI_H
