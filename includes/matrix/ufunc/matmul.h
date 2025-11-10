#ifndef MATRIX_UFUNC_MATMUL_H
#define MATRIX_UFUNC_MATMUL_H

#include "../viewer.h"
#include "add.h"
#include <assert.h>

#ifdef USE_MPI
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#endif

namespace ufunc {
namespace matmul {


template<class T>
class Seq {
public:
	static inline void operate(Buffer<T>& out, const Buffer<T>& lhs, const Buffer<T>& rhs) {
	#if MATMUL_ORDER == MATMUL_IJK
		#if PRECISION_MODE == 0
			for (size_t i = 0; i < out.rdim; i++) {
				for (size_t j = 0; j < out.cdim; j++) {
					T sum = 0;
					for (size_t k = 0; k < lhs.cdim; k++) {
						sum += lhs.data[i * lhs.cdim + k] * rhs.data[k * lhs.cdim + j];
					}
					out.data[i * out.cdim + j] = sum;
				}
			}
		#elif PRECISION_MODE == PRECISION_KANAN
			for (size_t i = 0; i < out.rdim; i++) {
				for (size_t j = 0; j < out.cdim; j++) {
					T sum = 0, c = 0;
					for (size_t k = 0; k < lhs.cdim; k++) {
						T y = lhs.data[i * lhs.cdim + k] * rhs.data[k * lhs.cdim + j] - c;
						T t = sum + y;
						c = (t - sum) - y;
						sum = t;
					}
					out.data[i * out.cdim + j] = sum;
				}
			}
		#elif PRECISION_MODE == PRECISION_NEUMAIER
			#error "PRECISION_MODE=PRECISION_NEUMAIER IS NOT IMPLEMENTED"
		#else  // #ifndef PRECISION_MODE
			#error "THIS PRECISION_MODE IS NOT IMPLEMENTED"
		#endif // #ifndef PRECISION_MODE
	#else // #if not def IJK_ORDER
		#if PRECISION_MODE == 0
			for (size_t i = 0; i < out.rdim; i++) {
					for (size_t k = 0; k < lhs.cdim; k++) {
						const T lhs_val = lhs.data[i * lhs.cdim + k];
						for (size_t j = 0; j < out.cdim; j++) {
							const T rhs_val = rhs.data[k * rhs.cdim + j];
							out.data[i * out.cdim + j] += lhs_val * rhs_val;
						}
					}
				}
		#else // #if def PRECISION_MODE
			#error "MATMUL IKJ_ORDER with PRECISION_MODE is not implemented"
		#endif // #ifndef PRECISION_MODE
	#endif // #if MATMUL_ORDER == MATMUL_IJK
	}

	static inline void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) {
	#if MATMUL_ORDER == MATMUL_IJK
		#if PRECISION_MODE == 0
			for (size_t i = 0; i < out.rdim; i++) {
				for (size_t j = 0; j < out.cdim; j++) {
					const size_t out_idx = out.map2Dto1DIndex(i, j);

					T sum = 0;
					for (size_t k = 0; k < lhs.cdim; k++) {
						const size_t lhs_idx = lhs.map2Dto1DIndex(i, k);
						const size_t rhs_idx = rhs.map2Dto1DIndex(k, j);
						sum += lhs.get(lhs_idx) * rhs.get(rhs_idx);
					}
					out.set(out_idx, sum);
				}
			}
		#elif PRECISION_MODE == PRECISION_KANAN
			for (size_t i = 0; i < out.rdim; i++) {
				for (size_t j = 0; j < out.cdim; j++) {
					const size_t out_idx = out.map2Dto1DIndex(i, j);

					T sum = 0, c = 0;
					for (size_t k = 0; k < lhs.cdim; k++) {
						const size_t lhs_idx = lhs.map2Dto1DIndex(i, k);
						const size_t rhs_idx = rhs.map2Dto1DIndex(k, j);

						const T prod = lhs.get(lhs_idx) * rhs.get(rhs_idx);
						const T y = prod - c;
						const T t = sum + y;
						c = (t - sum) - y;
						sum = t;
					}

					out.set(out_idx, sum);
				}
			}
		#elif PRECISION_MODE == PRECISION_NEUMAIER
			#error "PRECISION_MODE=PRECISION_NEUMAIER IS NOT IMPLEMENTED"
		#else  // #ifndef PRECISION_MODE
			#error "THIS PRECISION_MODE IS NOT IMPLEMENTED"
		#endif // #ifndef PRECISION_MODE
	#else // #if not def IJK_ORDER
		#if PRECISION_MODE == 0
			for (size_t i = 0; i < out.rdim; i++) {
				for (size_t k = 0; k < lhs.cdim; k++) {
					const size_t lhs_idx = lhs.map2Dto1DIndex(i, k);
					const T lhs_val = lhs.get(lhs_idx);
					for (size_t j = 0; j < out.cdim; j++) {
						const size_t rhs_idx = rhs.map2Dto1DIndex(k, j);
						const size_t out_idx = out.map2Dto1DIndex(i, j);

						const T rhs_val = rhs.get(rhs_idx);
						const T prev_out = out.get(out_idx);

						out.set(out_idx, prev_out + lhs_val * rhs_val);
					}
				}
			}
		#else // #if def PRECISION_MODE
			#error "MATMUL IKJ_ORDER with PRECISION_MODE is not implemented"
		#endif // #ifndef PRECISION_MODE
	#endif // #if MATMUL_ORDER == MATMUL_IJK
	}
};


template<class T>
class OmpVanilla {
public:
	static inline void operate(Buffer<T>& out, const Buffer<T>& lhs, const Buffer<T>& rhs) {
		#pragma omp parallel for
		for (size_t i = 0; i < out.rdim; i++) {
			for (size_t k = 0; k < lhs.cdim; k++) {
				const T lhs_val = lhs.data[i * lhs.cdim + k];
				for (size_t j = 0; j < out.rdim; j++) {
					const T rhs_val = rhs.data[k * rhs.cdim + j];
					out[i * out.cdim + j] += lhs_val * rhs_val;
				}
			}
		}
	}

	static inline void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) {
		#pragma omp parallel for
		for (size_t i = 0; i < out.rdim; i++) {
			for (size_t k = 0; k < lhs.cdim; k++) {
				const size_t lhs_idx = lhs.map2Dto1DIndex(i, k);
				const T lhs_val = lhs.get(lhs_idx);
				for (size_t j = 0; j < out.cdim; j++) {
					const size_t rhs_idx = rhs.map2Dto1DIndex(k, j);
					const size_t out_idx = out.map2Dto1DIndex(i, j);
					const T rhs_val = rhs.get(rhs_idx);
					const T prev_out = out.get(out_idx);
					
					out.set(out_idx, prev_out + lhs_val * rhs_val);
				}
			}
		}
	}
};


template<class T>
class OmpForkJoin {
public:
	static inline void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) {
		#pragma omp parallel
		#pragma omp single
		compute(out, lhs, rhs
			CODE_FOR_DEBUG_MODE(, "")
		);
	}

	static void set_threshold(size_t _threshold) {
		threshold = _threshold;
	}

protected:
	static size_t threshold;
	SplittableMatrix<T>* out;
	const SplittableMatrix<T>* lhs;
	const SplittableMatrix<T>* rhs;

protected:
	OmpForkJoin(SplittableMatrix<T>* out, const SplittableMatrix<T>* lhs, const SplittableMatrix<T>* rhs)
		: out(out), lhs(lhs), rhs(rhs) {}
	
	static inline void compute(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs
		CODE_FOR_DEBUG_MODE(, std::string format)
	) {
		CODE_FOR_DEBUG_MODE(printf(
			"[ufunc][matmul][OmpForkJoin] tid = \033[1;95m%d\033[0m " "format = \033[1;95m%-5s\033[0m, " "dim = \033[1;95m%6zu\033[0m, " "rDis = \033[1;95m%6zu\033[0m, " "cDis = \033[1;95m%6zu\033[0m\n", 
			omp_get_thread_num(), format.c_str(), out.rdim, out.rDis / out.rdim, out.cDis / out.cdim);
		)
		
		const size_t N = out.cdim;
		if (N <= threshold) {
			Seq<T>::operate(out, lhs, rhs);
			return;
		}

		SplittableMatrix<T> temp[2] = {
			SplittableMatrix<T>(new Buffer<T>(N), N),
			SplittableMatrix<T>(new Buffer<T>(N), N)
		};

		OmpForkJoin<T>* tasks[8];

		for (size_t i = 0; i < 2; i++) {
			for (size_t j = 0; j < 2; j++) {
				for (size_t k = 0; k < 2; k++) {
					// The following loop must iterate backward.
					// This is because Nguyenpanda designed the ForkJoin model to execute tasks
					// from left to right and from top to bottom.
					tasks[7 - (4 * i + 2 * j + k)] = new OmpForkJoin<T>(
						temp[i].split(j, k),
						lhs.split(j, i),
						rhs.split(i, k)
					);
				}
			}
		}

		for (size_t i = 0; i < 8; i++) {
			CODE_FOR_DEBUG_MODE(std::string fmt_str = format + std::to_string(i);)
			#pragma omp task
			compute(*tasks[i]->out, *tasks[i]->lhs, *tasks[i]->rhs 
				CODE_FOR_DEBUG_MODE(, fmt_str)
			);
		}

		#pragma omp taskwait

		addition::OmpForkJoin<T>::operate(out, temp[0], temp[1]);

		delete temp[0].root;
		delete temp[1].root;
		
		for (int i = 0; i < 8; i++) {
			delete tasks[i]->lhs;
			delete tasks[i]->rhs;
			delete tasks[i]->out;
			delete tasks[i];
		}
	}
};

template<class T> 
size_t OmpForkJoin<T>::threshold = -1;


#ifdef USE_MPI
// Baseline MPI implementation using simple row distribution and IKJ loop order
template<class T>
class MPIBaseline {
public:
	static void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) {
		int rank, size;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);

		int n = static_cast<int>(out.rdim);
		int workers = size - 1;

		if (workers <= 0) {
			if (rank == 0) {
				fprintf(stderr, "Error: Need at least 2 processes (1 master + workers)\n");
			}
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		if (rank == 0) {
			rootProcess(out, lhs, rhs, n, workers);
		} else {
			workerProcess(lhs, rhs, n);
		}
	}

protected:
	static void rootProcess(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, 
	                          const SplittableMatrix<T>& B, int n, int workers) {
		int rowsPerWorker = n / workers;
		int remainder = n % workers;

		MPI_Datatype mpi_type = sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE;

		// Send row blocks of A to each worker
		int start = 0;
		for (int p = 1; p <= workers; ++p) {
			int rows = rowsPerWorker + (p <= remainder ? 1 : 0);
			int end = start + rows;

			// Send start and end row indices
			MPI_Send(&start, 1, MPI_INT, p, 1, MPI_COMM_WORLD);
			MPI_Send(&end, 1, MPI_INT, p, 2, MPI_COMM_WORLD);

			// Send rows of A (use contiguous data from root buffer)
			T* A_start = A.root->data + start * n;
			MPI_Send(A_start, rows * n, mpi_type, p, 0, MPI_COMM_WORLD);

			start = end;
		}

		// Master also participates in broadcast of B
		MPI_Bcast(B.root->data, n * n, mpi_type, 0, MPI_COMM_WORLD);

		// Receive computed rows back from workers
		start = 0;
		for (int p = 1; p <= workers; ++p) {
			int rows = rowsPerWorker + (p <= remainder ? 1 : 0);
			
			T* C_start = C.root->data + start * n;
			MPI_Recv(C_start, rows * n, mpi_type, p, 3, 
			        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			start += rows;
		}
	}

	static void workerProcess(const SplittableMatrix<T>& A, const SplittableMatrix<T>& B, int n) {
		int startRow, endRow;
		MPI_Datatype mpi_type = sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE;

		// Receive start and end row indices
		MPI_Recv(&startRow, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&endRow, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		int rows = endRow - startRow;

		// Receive assigned rows of A
		Buffer<T>* A_buffer = new Buffer<T>(rows, n);
		MPI_Recv(A_buffer->data, rows * n, mpi_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// Receive entire matrix B via broadcast
		Buffer<T>* B_buffer = new Buffer<T>(n, n);
		MPI_Bcast(B_buffer->data, n * n, mpi_type, 0, MPI_COMM_WORLD);

		// Allocate result buffer
		Buffer<T>* C_buffer = new Buffer<T>(rows, n);
		for (int i = 0; i < rows * n; ++i) {
			C_buffer->data[i] = static_cast<T>(0);
		}

		// Compute using IKJ loop order (cache-friendly)
		for (int i = 0; i < rows; ++i) {
			for (int k = 0; k < n; ++k) {
				T aik = A_buffer->data[i * n + k];
				for (int j = 0; j < n; ++j) {
					C_buffer->data[i * n + j] += aik * B_buffer->data[k * n + j];
				}
			}
		}

		// Send computed rows back to master
		MPI_Send(C_buffer->data, rows * n, mpi_type, 0, 3, MPI_COMM_WORLD);

		delete A_buffer;
		delete B_buffer;
		delete C_buffer;
	}
};


// Hybrid Baseline: Row-distribution MPI + OpenMP parallelization
// Combines simple master-worker pattern with OpenMP parallel loops
template<class T>
class HybridBaseline {
public:
	static void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) {
		int rank, size;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);

		int n = static_cast<int>(out.rdim);
		int workers = size - 1;

		if (workers <= 0) {
			if (rank == 0) {
				fprintf(stderr, "Error: Need at least 2 processes (1 master + workers)\n");
			}
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		if (rank == 0) {
			rootProcess(out, lhs, rhs, n, workers);
		} else {
			workerProcess(lhs, rhs, n);
		}
	}

protected:
	static void rootProcess(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, 
	                          const SplittableMatrix<T>& B, int n, int workers) {
		int rowsPerWorker = n / workers;
		int remainder = n % workers;

		MPI_Datatype mpi_type = sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE;

		// Send row blocks of A to each worker
		int start = 0;
		for (int p = 1; p <= workers; ++p) {
			int rows = rowsPerWorker + (p <= remainder ? 1 : 0);
			int end = start + rows;

			// Send start and end row indices
			MPI_Send(&start, 1, MPI_INT, p, 1, MPI_COMM_WORLD);
			MPI_Send(&end, 1, MPI_INT, p, 2, MPI_COMM_WORLD);

			// Send rows of A (use contiguous data from root buffer)
			T* A_start = A.root->data + start * n;
			MPI_Send(A_start, rows * n, mpi_type, p, 0, MPI_COMM_WORLD);

			start = end;
		}

		// Master also participates in broadcast of B
		MPI_Bcast(B.root->data, n * n, mpi_type, 0, MPI_COMM_WORLD);

		// Receive computed rows back from workers
		start = 0;
		for (int p = 1; p <= workers; ++p) {
			int rows = rowsPerWorker + (p <= remainder ? 1 : 0);
			
			T* C_start = C.root->data + start * n;
			MPI_Recv(C_start, rows * n, mpi_type, p, 3, 
			        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			start += rows;
		}
	}

	static void workerProcess(const SplittableMatrix<T>& A, const SplittableMatrix<T>& B, int n) {
		int startRow, endRow;
		MPI_Datatype mpi_type = sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE;

		// Receive start and end row indices
		MPI_Recv(&startRow, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&endRow, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		int rows = endRow - startRow;

		// Receive assigned rows of A
		Buffer<T>* A_buffer = new Buffer<T>(rows, n);
		MPI_Recv(A_buffer->data, rows * n, mpi_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// Receive entire matrix B via broadcast
		Buffer<T>* B_buffer = new Buffer<T>(n, n);
		MPI_Bcast(B_buffer->data, n * n, mpi_type, 0, MPI_COMM_WORLD);

		// Allocate result buffer
		Buffer<T>* C_buffer = new Buffer<T>(rows, n);
		for (int i = 0; i < rows * n; ++i) {
			C_buffer->data[i] = static_cast<T>(0);
		}

		// Compute using IKJ loop order with OpenMP parallelization
		#pragma omp parallel for
		for (int i = 0; i < rows; ++i) {
			for (int k = 0; k < n; ++k) {
				T aik = A_buffer->data[i * n + k];
				for (int j = 0; j < n; ++j) {
					C_buffer->data[i * n + j] += aik * B_buffer->data[k * n + j];
				}
			}
		}

		// Send computed rows back to master
		MPI_Send(C_buffer->data, rows * n, mpi_type, 0, 3, MPI_COMM_WORLD);

		delete A_buffer;
		delete B_buffer;
		delete C_buffer;
	}
};


template<class T>
class MPIGridForkJoin {
public:
	static const int THRESHOLD = 64;

	static void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) {
		int rank, size;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);

		int grid_size = static_cast<int>(std::sqrt(size));
		if (grid_size * grid_size != size) {
			if (rank == 0) {
				fprintf(stderr, "Error: Number of processes must be a perfect square\n");
			}
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		int n = static_cast<int>(out.rdim);
		
		if (rank == 0) {
			rootProcess(out, lhs, rhs, n, grid_size);
		} else {
			workerProcess(n, grid_size, rank);
		}
	}

protected:
	static void multiplyRecursive(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, 
	                              const SplittableMatrix<T>& B) {
		int m = static_cast<int>(C.rdim);
		int k = static_cast<int>(A.cdim);
		int n = static_cast<int>(C.cdim);
		
		if (m <= THRESHOLD && k <= THRESHOLD && n <= THRESHOLD) {
			// Base case: direct multiplication using SplittableMatrix
			for (int i = 0; i < m; ++i) {
				for (int kk = 0; kk < k; ++kk) {
					size_t a_idx = A.map2Dto1DIndex(i, kk);
					T aik = A.get(a_idx);
					for (int j = 0; j < n; ++j) {
						size_t b_idx = B.map2Dto1DIndex(kk, j);
						size_t c_idx = C.map2Dto1DIndex(i, j);
						C.set(c_idx, C.get(c_idx) + aik * B.get(b_idx));
					}
				}
			}
			return;
		}

		// Recursive case: divide and conquer using views
		int half_m = m / 2;
		int half_k = k / 2;
		int half_n = n / 2;
		int rem_m = m - half_m;
		int rem_k = k - half_k;
		int rem_n = n - half_n;

		// C00 = A00*B00 + A01*B10
		SplittableMatrix<T>* C00 = C.view(0, 0, half_m, half_n);
		SplittableMatrix<T>* A00 = A.view(0, 0, half_m, half_k);
		SplittableMatrix<T>* B00 = B.view(0, 0, half_k, half_n);
		multiplyRecursive(*C00, *A00, *B00);
		delete C00; delete A00; delete B00;

		SplittableMatrix<T>* C00_2 = C.view(0, 0, half_m, half_n);
		SplittableMatrix<T>* A01 = A.view(0, half_k, half_m, rem_k);
		SplittableMatrix<T>* B10 = B.view(half_k, 0, rem_k, half_n);
		multiplyRecursive(*C00_2, *A01, *B10);
		delete C00_2; delete A01; delete B10;

		// C01 = A00*B01 + A01*B11
		SplittableMatrix<T>* C01 = C.view(0, half_n, half_m, rem_n);
		SplittableMatrix<T>* A00_2 = A.view(0, 0, half_m, half_k);
		SplittableMatrix<T>* B01 = B.view(0, half_n, half_k, rem_n);
		multiplyRecursive(*C01, *A00_2, *B01);
		delete C01; delete A00_2; delete B01;

		SplittableMatrix<T>* C01_2 = C.view(0, half_n, half_m, rem_n);
		SplittableMatrix<T>* A01_2 = A.view(0, half_k, half_m, rem_k);
		SplittableMatrix<T>* B11 = B.view(half_k, half_n, rem_k, rem_n);
		multiplyRecursive(*C01_2, *A01_2, *B11);
		delete C01_2; delete A01_2; delete B11;

		// C10 = A10*B00 + A11*B10
		SplittableMatrix<T>* C10 = C.view(half_m, 0, rem_m, half_n);
		SplittableMatrix<T>* A10 = A.view(half_m, 0, rem_m, half_k);
		SplittableMatrix<T>* B00_2 = B.view(0, 0, half_k, half_n);
		multiplyRecursive(*C10, *A10, *B00_2);
		delete C10; delete A10; delete B00_2;

		SplittableMatrix<T>* C10_2 = C.view(half_m, 0, rem_m, half_n);
		SplittableMatrix<T>* A11 = A.view(half_m, half_k, rem_m, rem_k);
		SplittableMatrix<T>* B10_2 = B.view(half_k, 0, rem_k, half_n);
		multiplyRecursive(*C10_2, *A11, *B10_2);
		delete C10_2; delete A11; delete B10_2;

		// C11 = A10*B01 + A11*B11
		SplittableMatrix<T>* C11 = C.view(half_m, half_n, rem_m, rem_n);
		SplittableMatrix<T>* A10_2 = A.view(half_m, 0, rem_m, half_k);
		SplittableMatrix<T>* B01_2 = B.view(0, half_n, half_k, rem_n);
		multiplyRecursive(*C11, *A10_2, *B01_2);
		delete C11; delete A10_2; delete B01_2;

		SplittableMatrix<T>* C11_2 = C.view(half_m, half_n, rem_m, rem_n);
		SplittableMatrix<T>* A11_2 = A.view(half_m, half_k, rem_m, rem_k);
		SplittableMatrix<T>* B11_2 = B.view(half_k, half_n, rem_k, rem_n);
		multiplyRecursive(*C11_2, *A11_2, *B11_2);
		delete C11_2; delete A11_2; delete B11_2;
	}

	static void rootProcess(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, 
	                        const SplittableMatrix<T>& B, int n, int grid_size) {
		// Calculate block sizes
		int base_block_rows = n / grid_size;
		int base_block_cols = n / grid_size;
		int extra_rows = n % grid_size;
		int extra_cols = n % grid_size;

		MPI_Datatype mpi_type = sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE;

		// Create MPI subarray datatypes for all blocks (zero-copy views)
		std::vector<std::vector<SplittableMatrix<T>*>> A_views(grid_size, 
			std::vector<SplittableMatrix<T>*>(grid_size));
		std::vector<std::vector<SplittableMatrix<T>*>> B_views(grid_size, 
			std::vector<SplittableMatrix<T>*>(grid_size));

		for (int grid_row = 0; grid_row < grid_size; ++grid_row) {
			int num_rows = base_block_rows + (grid_row < extra_rows ? 1 : 0);
			int row_start = grid_row * base_block_rows + std::min(grid_row, extra_rows);

			for (int grid_col = 0; grid_col < grid_size; ++grid_col) {
				int num_cols = base_block_cols + (grid_col < extra_cols ? 1 : 0);
				int col_start = grid_col * base_block_cols + std::min(grid_col, extra_cols);

				// Create zero-copy views (just pointer arithmetic, no data copy)
				A_views[grid_row][grid_col] = A.view(row_start, col_start, num_rows, num_cols);
				B_views[grid_row][grid_col] = B.view(row_start, col_start, num_rows, num_cols);
			}
		}

		// Send blocks to workers using MPI derived datatypes
		std::vector<MPI_Request> send_requests;
		std::vector<MPI_Datatype> datatypes; // Keep track for cleanup

		for (int dest_rank = 1; dest_rank < grid_size * grid_size; ++dest_rank) {
			int dest_row = dest_rank / grid_size;
			int dest_col = dest_rank % grid_size;

			for (int k_block = 0; k_block < grid_size; ++k_block) {
				SplittableMatrix<T>* A_block = A_views[dest_row][k_block];
				SplittableMatrix<T>* B_block = B_views[k_block][dest_col];

				// Create MPI subarray datatype for non-contiguous A block
				MPI_Datatype A_subarray;
				int A_sizes[2] = {(int)A.root->rdim, (int)A.root->cdim};
				int A_subsizes[2] = {(int)A_block->rdim, (int)A_block->cdim};
				int A_starts[2] = {(int)A_block->rDis, (int)A_block->cDis};
				MPI_Type_create_subarray(2, A_sizes, A_subsizes, A_starts, 
				                         MPI_ORDER_C, mpi_type, &A_subarray);
				MPI_Type_commit(&A_subarray);
				datatypes.push_back(A_subarray);

				send_requests.push_back(MPI_Request());
				MPI_Isend(A.root->data, 1, A_subarray, dest_rank, 
				         k_block * 2, MPI_COMM_WORLD, &send_requests.back());

				// Create MPI subarray datatype for non-contiguous B block
				MPI_Datatype B_subarray;
				int B_sizes[2] = {(int)B.root->rdim, (int)B.root->cdim};
				int B_subsizes[2] = {(int)B_block->rdim, (int)B_block->cdim};
				int B_starts[2] = {(int)B_block->rDis, (int)B_block->cDis};
				MPI_Type_create_subarray(2, B_sizes, B_subsizes, B_starts, 
				                         MPI_ORDER_C, mpi_type, &B_subarray);
				MPI_Type_commit(&B_subarray);
				datatypes.push_back(B_subarray);

				send_requests.push_back(MPI_Request());
				MPI_Isend(B.root->data, 1, B_subarray, dest_rank, 
				         k_block * 2 + 1, MPI_COMM_WORLD, &send_requests.back());
			}
		}

		// Compute local result (rank 0's contribution)
		int local_rows = base_block_rows + (0 < extra_rows ? 1 : 0);
		int local_cols = base_block_cols + (0 < extra_cols ? 1 : 0);
		
		// Create output view for local result
		SplittableMatrix<T>* C_local = C.view(0, 0, local_rows, local_cols);

		for (int k_block = 0; k_block < grid_size; ++k_block) {
			SplittableMatrix<T>* A_local = A_views[0][k_block];
			SplittableMatrix<T>* B_local = B_views[k_block][0];
			
			// Compute directly using views (zero-copy!)
			multiplyRecursive(*C_local, *A_local, *B_local);
		}
		
		delete C_local;

		// Receive results from workers
		for (int src_rank = 1; src_rank < grid_size * grid_size; ++src_rank) {
			int src_row = src_rank / grid_size;
			int src_col = src_rank % grid_size;
			int result_rows = base_block_rows + (src_row < extra_rows ? 1 : 0);
			int result_cols = base_block_cols + (src_col < extra_cols ? 1 : 0);
			int row_start = src_row * base_block_rows + std::min(src_row, extra_rows);
			int col_start = src_col * base_block_cols + std::min(src_col, extra_cols);

			std::vector<T> result_block(result_rows * result_cols);
			MPI_Recv(result_block.data(), result_rows * result_cols,
			        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, src_rank, 0,
			        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// Insert result using zero-copy view
			SplittableMatrix<T>* C_result = C.view(row_start, col_start, result_rows, result_cols);
			for (int i = 0; i < result_rows; ++i) {
				for (int j = 0; j < result_cols; ++j) {
					C_result->set(C_result->map2Dto1DIndex(i, j), result_block[i * result_cols + j]);
				}
			}
			delete C_result;
		}

		MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);

		// Cleanup MPI datatypes
		for (auto& dtype : datatypes) {
			MPI_Type_free(&dtype);
		}

		// Cleanup views
		for (int i = 0; i < grid_size; ++i) {
			for (int j = 0; j < grid_size; ++j) {
				delete A_views[i][j];
				delete B_views[i][j];
			}
		}
	}

	static void workerProcess(int n, int grid_size, int rank) {
		int my_row = rank / grid_size;
		int my_col = rank % grid_size;

		int base_block_rows = n / grid_size;
		int base_block_cols = n / grid_size;
		int extra_rows = n % grid_size;
		int extra_cols = n % grid_size;

		int my_rows = base_block_rows + (my_row < extra_rows ? 1 : 0);
		int my_cols = base_block_cols + (my_col < extra_cols ? 1 : 0);

		// Create result buffer and SplittableMatrix wrapper
		Buffer<T>* result_buffer = new Buffer<T>(my_rows, my_cols);
		for (int i = 0; i < my_rows * my_cols; ++i) {
			result_buffer->data[i] = static_cast<T>(0);
		}
		SplittableMatrix<T> result(result_buffer, my_rows, my_cols);

		for (int k_block = 0; k_block < grid_size; ++k_block) {
			int k_size = base_block_cols + (k_block < extra_cols ? 1 : 0);
			int a_rows = my_rows;
			int a_cols = k_size;
			int b_rows = k_size;
			int b_cols = my_cols;

			// Receive A and B blocks into buffers
			Buffer<T>* A_buffer = new Buffer<T>(a_rows, a_cols);
			Buffer<T>* B_buffer = new Buffer<T>(b_rows, b_cols);

			MPI_Recv(A_buffer->data, a_rows * a_cols,
			        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, 0, k_block * 2,
			        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			MPI_Recv(B_buffer->data, b_rows * b_cols,
			        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, 0, k_block * 2 + 1,
			        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// Wrap in SplittableMatrix and compute using views
			SplittableMatrix<T> A_block(A_buffer, a_rows, a_cols);
			SplittableMatrix<T> B_block(B_buffer, b_rows, b_cols);
			
			multiplyRecursive(result, A_block, B_block);
			
			delete A_buffer;
			delete B_buffer;
		}

		MPI_Send(result_buffer->data, my_rows * my_cols,
		        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		
		delete result_buffer;
	}
};


template<class T>
class HybridGrid {
public:
	static const int THRESHOLD = 64;

	static void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) {
		int rank, size;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);

		int grid_size = static_cast<int>(std::sqrt(size));
		if (grid_size * grid_size != size) {
			if (rank == 0) {
				fprintf(stderr, "Error: Number of processes must be a perfect square\n");
			}
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		int n = static_cast<int>(out.rdim);
		
		if (rank == 0) {
			rootProcess(out, lhs, rhs, n, grid_size);
		} else {
			workerProcess(n, grid_size, rank);
		}
	}

protected:
	// Hybrid recursive multiply: combines OpenMP task parallelism with recursive divide-and-conquer
	static void multiplyRecursive(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, 
	                              const SplittableMatrix<T>& B) {
		int m = static_cast<int>(C.rdim);
		int k = static_cast<int>(A.cdim);
		int n = static_cast<int>(C.cdim);
		
		if (m <= THRESHOLD && k <= THRESHOLD && n <= THRESHOLD) {
			// Base case: direct multiplication with OpenMP parallel for
			#pragma omp parallel for
			for (int i = 0; i < m; ++i) {
				for (int kk = 0; kk < k; ++kk) {
					size_t a_idx = A.map2Dto1DIndex(i, kk);
					T aik = A.get(a_idx);
					for (int j = 0; j < n; ++j) {
						size_t b_idx = B.map2Dto1DIndex(kk, j);
						size_t c_idx = C.map2Dto1DIndex(i, j);
						C.set(c_idx, C.get(c_idx) + aik * B.get(b_idx));
					}
				}
			}
			return;
		}

		// Recursive case: divide and conquer with OpenMP tasks
		int half_m = m / 2;
		int half_k = k / 2;
		int half_n = n / 2;
		int rem_m = m - half_m;
		int rem_k = k - half_k;
		int rem_n = n - half_n;

		// Create all submatrix views
		SplittableMatrix<T>* A00 = A.view(0, 0, half_m, half_k);
		SplittableMatrix<T>* A01 = A.view(0, half_k, half_m, rem_k);
		SplittableMatrix<T>* A10 = A.view(half_m, 0, rem_m, half_k);
		SplittableMatrix<T>* A11 = A.view(half_m, half_k, rem_m, rem_k);

		SplittableMatrix<T>* B00 = B.view(0, 0, half_k, half_n);
		SplittableMatrix<T>* B01 = B.view(0, half_n, half_k, rem_n);
		SplittableMatrix<T>* B10 = B.view(half_k, 0, rem_k, half_n);
		SplittableMatrix<T>* B11 = B.view(half_k, half_n, rem_k, rem_n);

		SplittableMatrix<T>* C00 = C.view(0, 0, half_m, half_n);
		SplittableMatrix<T>* C01 = C.view(0, half_n, half_m, rem_n);
		SplittableMatrix<T>* C10 = C.view(half_m, 0, rem_m, half_n);
		SplittableMatrix<T>* C11 = C.view(half_m, half_n, rem_m, rem_n);

		#pragma omp parallel
		{
			#pragma omp single
			{
				// C00 = A00*B00 + A01*B10
				#pragma omp task shared(C00, A00, B00)
				{
					multiplyRecursive(*C00, *A00, *B00);
				}
				#pragma omp task shared(C00, A01, B10)
				{
					multiplyRecursive(*C00, *A01, *B10);
				}

				// C01 = A00*B01 + A01*B11
				#pragma omp task shared(C01, A00, B01)
				{
					multiplyRecursive(*C01, *A00, *B01);
				}
				#pragma omp task shared(C01, A01, B11)
				{
					multiplyRecursive(*C01, *A01, *B11);
				}

				// C10 = A10*B00 + A11*B10
				#pragma omp task shared(C10, A10, B00)
				{
					multiplyRecursive(*C10, *A10, *B00);
				}
				#pragma omp task shared(C10, A11, B10)
				{
					multiplyRecursive(*C10, *A11, *B10);
				}

				// C11 = A10*B01 + A11*B11
				#pragma omp task shared(C11, A10, B01)
				{
					multiplyRecursive(*C11, *A10, *B01);
				}
				#pragma omp task shared(C11, A11, B11)
				{
					multiplyRecursive(*C11, *A11, *B11);
				}

				#pragma omp taskwait
			}
		}

		// Clean up views
		delete A00; delete A01; delete A10; delete A11;
		delete B00; delete B01; delete B10; delete B11;
		delete C00; delete C01; delete C10; delete C11;
	}

	static void rootProcess(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, 
	                       const SplittableMatrix<T>& B, int n, int grid_size) {
		int base_block_rows = n / grid_size;
		int base_block_cols = n / grid_size;
		int extra_rows = n % grid_size;
		int extra_cols = n % grid_size;

		MPI_Datatype mpi_type = sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE;

		// Create views for all blocks
		std::vector<std::vector<SplittableMatrix<T>*>> A_views(grid_size, 
			std::vector<SplittableMatrix<T>*>(grid_size));
		std::vector<std::vector<SplittableMatrix<T>*>> B_views(grid_size, 
			std::vector<SplittableMatrix<T>*>(grid_size));

		for (int grid_row = 0; grid_row < grid_size; ++grid_row) {
			int num_rows = base_block_rows + (grid_row < extra_rows ? 1 : 0);
			int row_start = grid_row * base_block_rows + std::min(grid_row, extra_rows);

			for (int grid_col = 0; grid_col < grid_size; ++grid_col) {
				int num_cols = base_block_cols + (grid_col < extra_cols ? 1 : 0);
				int col_start = grid_col * base_block_cols + std::min(grid_col, extra_cols);

				A_views[grid_row][grid_col] = A.view(row_start, col_start, num_rows, num_cols);
				B_views[grid_row][grid_col] = B.view(row_start, col_start, num_rows, num_cols);
			}
		}

		// Send blocks to workers using MPI derived datatypes
		std::vector<MPI_Request> send_requests;
		std::vector<MPI_Datatype> datatypes;

		for (int dest_rank = 1; dest_rank < grid_size * grid_size; ++dest_rank) {
			int dest_row = dest_rank / grid_size;
			int dest_col = dest_rank % grid_size;

			for (int k_block = 0; k_block < grid_size; ++k_block) {
				SplittableMatrix<T>* A_block = A_views[dest_row][k_block];
				SplittableMatrix<T>* B_block = B_views[k_block][dest_col];

				// Create MPI subarray datatype for A block
				MPI_Datatype A_subarray;
				int A_sizes[2] = {(int)A.root->rdim, (int)A.root->cdim};
				int A_subsizes[2] = {(int)A_block->rdim, (int)A_block->cdim};
				int A_starts[2] = {(int)A_block->rDis, (int)A_block->cDis};
				MPI_Type_create_subarray(2, A_sizes, A_subsizes, A_starts, 
				                         MPI_ORDER_C, mpi_type, &A_subarray);
				MPI_Type_commit(&A_subarray);
				datatypes.push_back(A_subarray);

				send_requests.push_back(MPI_Request());
				MPI_Isend(A.root->data, 1, A_subarray, dest_rank, 
				         k_block * 2, MPI_COMM_WORLD, &send_requests.back());

				// Create MPI subarray datatype for B block
				MPI_Datatype B_subarray;
				int B_sizes[2] = {(int)B.root->rdim, (int)B.root->cdim};
				int B_subsizes[2] = {(int)B_block->rdim, (int)B_block->cdim};
				int B_starts[2] = {(int)B_block->rDis, (int)B_block->cDis};
				MPI_Type_create_subarray(2, B_sizes, B_subsizes, B_starts, 
				                         MPI_ORDER_C, mpi_type, &B_subarray);
				MPI_Type_commit(&B_subarray);
				datatypes.push_back(B_subarray);

				send_requests.push_back(MPI_Request());
				MPI_Isend(B.root->data, 1, B_subarray, dest_rank, 
				         k_block * 2 + 1, MPI_COMM_WORLD, &send_requests.back());
			}
		}

		// Compute local result (rank 0's contribution) with OpenMP
		int local_rows = base_block_rows + (0 < extra_rows ? 1 : 0);
		int local_cols = base_block_cols + (0 < extra_cols ? 1 : 0);
		
		SplittableMatrix<T>* C_local = C.view(0, 0, local_rows, local_cols);

		for (int k_block = 0; k_block < grid_size; ++k_block) {
			SplittableMatrix<T>* A_local = A_views[0][k_block];
			SplittableMatrix<T>* B_local = B_views[k_block][0];
			
			// Use hybrid recursive multiply with OpenMP
			multiplyRecursive(*C_local, *A_local, *B_local);
		}
		
		delete C_local;

		// Receive results from workers
		for (int src_rank = 1; src_rank < grid_size * grid_size; ++src_rank) {
			int src_row = src_rank / grid_size;
			int src_col = src_rank % grid_size;
			int result_rows = base_block_rows + (src_row < extra_rows ? 1 : 0);
			int result_cols = base_block_cols + (src_col < extra_cols ? 1 : 0);
			int row_start = src_row * base_block_rows + std::min(src_row, extra_rows);
			int col_start = src_col * base_block_cols + std::min(src_col, extra_cols);

			std::vector<T> result_block(result_rows * result_cols);
			MPI_Recv(result_block.data(), result_rows * result_cols,
			        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, src_rank, 0,
			        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			SplittableMatrix<T>* C_result = C.view(row_start, col_start, result_rows, result_cols);
			for (int i = 0; i < result_rows; ++i) {
				for (int j = 0; j < result_cols; ++j) {
					C_result->set(C_result->map2Dto1DIndex(i, j), result_block[i * result_cols + j]);
				}
			}
			delete C_result;
		}

		MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);

		// Cleanup
		for (auto& dtype : datatypes) {
			MPI_Type_free(&dtype);
		}

		for (int i = 0; i < grid_size; ++i) {
			for (int j = 0; j < grid_size; ++j) {
				delete A_views[i][j];
				delete B_views[i][j];
			}
		}
	}

	static void workerProcess(int n, int grid_size, int rank) {
		int my_row = rank / grid_size;
		int my_col = rank % grid_size;

		int base_block_rows = n / grid_size;
		int base_block_cols = n / grid_size;
		int extra_rows = n % grid_size;
		int extra_cols = n % grid_size;

		int my_rows = base_block_rows + (my_row < extra_rows ? 1 : 0);
		int my_cols = base_block_cols + (my_col < extra_cols ? 1 : 0);

		// Create result buffer
		Buffer<T>* result_buffer = new Buffer<T>(my_rows, my_cols);
		for (int i = 0; i < my_rows * my_cols; ++i) {
			result_buffer->data[i] = static_cast<T>(0);
		}
		SplittableMatrix<T> result(result_buffer, my_rows, my_cols);

		for (int k_block = 0; k_block < grid_size; ++k_block) {
			int k_size = base_block_cols + (k_block < extra_cols ? 1 : 0);
			int a_rows = my_rows;
			int a_cols = k_size;
			int b_rows = k_size;
			int b_cols = my_cols;

			// Receive A and B blocks
			Buffer<T>* A_buffer = new Buffer<T>(a_rows, a_cols);
			Buffer<T>* B_buffer = new Buffer<T>(b_rows, b_cols);

			MPI_Recv(A_buffer->data, a_rows * a_cols, 
			        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, 0, k_block * 2,
			        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			MPI_Recv(B_buffer->data, b_rows * b_cols,
			        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, 0, k_block * 2 + 1,
			        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// Wrap in SplittableMatrix
			SplittableMatrix<T> A_mat(A_buffer, a_rows, a_cols);
			SplittableMatrix<T> B_mat(B_buffer, b_rows, b_cols);

			// Compute with OpenMP hybrid recursion
			multiplyRecursive(result, A_mat, B_mat);

			delete A_buffer;
			delete B_buffer;
		}

		// Send result back to root
		MPI_Send(result.root->data, my_rows * my_cols,
		        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

		delete result_buffer;
	}
};
#endif // USE_MPI


}; // namespace matmul
}; // namespace ufunc

#endif // MATRIX_UFUNC_MATMUL_H
