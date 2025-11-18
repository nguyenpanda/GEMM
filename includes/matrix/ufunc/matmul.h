// #ifndef MATRIX_UFUNC_MATMUL_H
#define MATRIX_UFUNC_MATMUL_H

#include "../viewer.h"
#include "add.h"
#include <assert.h>

#ifdef USE_MPI
#include <mpi.h>
#include <thread>
#include <queue>
#include <cmath>
#include <vector>
#include <algorithm>
#include <atomic>
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


template<class T>
class OmpStrassen {
public:
	static inline void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) 
	{
		#pragma omp parallel
		#pragma omp single
		StrassenRecursive(out, lhs, rhs);
	}

	static void set_threshold(size_t _threshold) 
	{
		threshold = _threshold;
	}

protected:
	static size_t threshold;
	SplittableMatrix<T>* out;
	const SplittableMatrix<T>* lhs;
	const SplittableMatrix<T>* rhs;

protected:
	OmpStrassen(SplittableMatrix<T>* out, const SplittableMatrix<T>* lhs, const SplittableMatrix<T>* rhs)
		: out(out), lhs(lhs), rhs(rhs) {}
	
	static void StrassenRecursive(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B) 
	{
        // const size_t N = std::max(C.rdim, C.cdim);
		const size_t N = C.cdim;
        if (N <= threshold) 
		{
            Seq<T>::operate(C, A, B);
            return;
        }

		size_t hsize = N / 2;

		SplittableMatrix<T>* Temp1 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T>* Temp2 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T>* res = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T>* C00 = C.split(0, 0);
		SplittableMatrix<T>* C01 = C.split(0, 1);
		SplittableMatrix<T>* C10 = C.split(1, 0);
		SplittableMatrix<T>* C11 = C.split(1, 1);

        #pragma omp taskgroup task_reduction(+: C00, C01, C10, C11)
		{
			#pragma omp task in_reduction(+:C00, C11)
			{	
				ufunc::addition::Seq<T>::operate(*Temp1, *A.split(0, 0), *A.split(1, 1));
				ufunc::addition::Seq<T>::operate(*Temp1, *B.split(0, 0), *B.split(1, 1));
				StrassenRecursive(*res, *Temp1, *Temp2);
				ufunc::addition::Seq<T>::operate(*C00, *C00, *res);
				ufunc::addition::Seq<T>::operate(*C11, *C11, *res);
			}
			#pragma omp task in_reduction(+:C10, C11)
			{
				ufunc::addition::Seq<T>::operate(*Temp1, *A.split(1, 0), *A.split(1, 1));
				StrassenRecursive(*res, *Temp1, *B.split(0, 0));
				ufunc::addition::Seq<T>::operate(*C10, *C10, *res);
				ufunc::addition::Seq<T>::re_operate(*C11, *C11, *res);
			}
			#pragma omp task in_reduction(+:C01, C11)
			{
				ufunc::addition::Seq<T>::re_operate(*Temp1, *B.split(0, 1), *B.split(1, 1));
				StrassenRecursive(*res, *A.split(0, 0), *Temp1);
				ufunc::addition::Seq<T>::operate(*C01, *C01, *res);
				ufunc::addition::Seq<T>::re_operate(*C11, *C11, *res);
			}
			#pragma omp task in_reduction(+:C00, C10)
			{
				ufunc::addition::Seq<T>::re_operate(*Temp1, *B.split(1, 0), *B.split(0, 0));
				StrassenRecursive(*res, *A.split(1, 1), *Temp1);
				ufunc::addition::Seq<T>::operate(*C00, *C00, *res);
				ufunc::addition::Seq<T>::operate(*C10, *C10, *res);
			}
			#pragma omp task in_reduction(+:C00, C01)
			{
				ufunc::addition::Seq<T>::re_operate(*Temp1, *A.split(0, 0), *A.split(0, 1));
				StrassenRecursive(*res, *Temp1, *B.split(1, 1));
				ufunc::addition::Seq<T>::re_operate(*C00, *C00, *res);
				ufunc::addition::Seq<T>::operate(*C01, *C01, *res);
			}
			#pragma omp task in_reduction(+:C11)
			{
				ufunc::addition::Seq<T>::re_operate(*Temp1, *A.split(1, 0), *A.split(0, 0));
				ufunc::addition::Seq<T>::operate(*Temp2, *B.split(0, 0), *B.split(0, 1));
				StrassenRecursive(*res, *Temp1, *Temp2);
				ufunc::addition::Seq<T>::operate(*C11, *C11, *res);
			}
			#pragma omp task in_reduction(+:C00)
			{
				ufunc::addition::Seq<T>::re_operate(*Temp1, *A.split(1, 0), *A.split(0, 0));
				ufunc::addition::Seq<T>::operate(*Temp2, *B.split(0, 0), *B.split(0, 1));
				StrassenRecursive(*res, *Temp1, *Temp2);
				ufunc::addition::Seq<T>::operate(*C00, *C00, *res);
			}
		}

		delete A00; delete A01; delete A10; delete A11;
		delete B00; delete B01; delete B10; delete B11;
		delete Temp1; delete Temp2; delete res;


	}
};


template<class T> 
size_t OmpStrassen<T>::threshold = -1;

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



template<class T>
class HybridStrassen {
public:
	static inline void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) 
	{
		int rank, nprocs; 
    	MPI_Comm_rank(MPI_COMM_WORLD,&rank); 
    	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
		{

			std::thread scheduler_thread(scheduler_loop, nprocs );

			StrassenRecursive(out, lhs,  rhs, 0);

			for (int i = 1; i < nprocs; ++i) 
				MPI_Send(NULL,0,MPI_BYTE,i,TAG_STOP,MPI_COMM_WORLD);
			g_stop.store(true);

			scheduler_thread.join();

		}
		else 
		{
			worker_loop(rank);
		}
	}

	static void set_threshold(size_t _threshold) 
	{
		threshold = _threshold;
	}

protected:
	static size_t threshold;
	SplittableMatrix<T>* out;
	const SplittableMatrix<T>* lhs;
	const SplittableMatrix<T>* rhs;

	static std::atomic<bool> g_stop;

	enum {
		TAG_TASK = 100, TAG_RESULT = 101, TAG_STOP = 102,
		TAG_REQ_WORKER = 103, TAG_ASSIGN = 104, TAG_WORKER_FREE = 105
	};

	struct task
	{
		int size;
		int caller;
		int depth;
	};

	HybridStrassen(SplittableMatrix<T>* out, const SplittableMatrix<T>* lhs, const SplittableMatrix<T>* rhs)
		: out(out), lhs(lhs), rhs(rhs) {}

	static void worker_loop(int myRank)
	{
		while (true)
		{
			int flag = 0;
			MPI_Status st;
			MPI_Iprobe(0, TAG_STOP, MPI_COMM_WORLD, &flag, &st);
			if (flag)
			{
				MPI_Recv(NULL, 0, MPI_BYTE, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				break;
			}
			
			MPI_Iprobe(MPI_ANY_SOURCE, TAG_TASK, MPI_COMM_WORLD, &flag, &st);
			if (!flag)
			{
				continue; 
			}

			task h;
			MPI_Recv(&h, 3, MPI_INT, st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			int n = h.size;

			SplittableMatrix<T> *A = new SplittableMatrix<T>(new Buffer<T>(n, n), n, n);
			SplittableMatrix<T> *B = new SplittableMatrix<T>(new Buffer<T>(n, n), n, n);

			MPI_Recv(A->root->data, n*n, sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(B->root->data, n*n, sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			StrassenRecursive(*A, *A, *B, h.depth);

			MPI_Send(&h, 3, MPI_INT, h.caller, TAG_RESULT, MPI_COMM_WORLD);
			MPI_Send(A->root->data, n*n, sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, h.caller, TAG_RESULT, MPI_COMM_WORLD);
			MPI_Send(&myRank, 1, MPI_INT, 0, TAG_WORKER_FREE, MPI_COMM_WORLD);

		}
	}

	static int request_worker(int depth) 
	{
		MPI_Send(&depth, 1, MPI_INT, 0, TAG_REQ_WORKER, MPI_COMM_WORLD);
		int worker; 
		MPI_Recv(&worker, 1, MPI_INT, 0, TAG_ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		return worker; 
	}

	static void send_data(SplittableMatrix<T>& a, SplittableMatrix<T>& b, int n, int depth, int worker, SplittableMatrix<T>& res)
	{
		task h {
			.size = n,
			.depth = depth
		};
		MPI_Comm_rank(MPI_COMM_WORLD, &h.caller);

		MPI_Send(&h, 3, MPI_INT, worker, TAG_TASK, MPI_COMM_WORLD);

		MPI_Send(a.root->data, n*n, sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, worker, TAG_TASK, MPI_COMM_WORLD);
		MPI_Send(b.root->data, n*n, sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, worker, TAG_TASK, MPI_COMM_WORLD);
		
		// nhận kết quả
		task rh;
		MPI_Recv(&rh, 3, MPI_INT, worker, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(res.root->data, rh.size*rh.size, sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, worker, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
	}

	static inline void scheduler_loop(int nprocs)
	{
		std::queue<int> free_workers;
		for (int w = 1; w < nprocs; ++w) 
			free_workers.push(w);

		MPI_Status st;

		while (!g_stop)
		{
			int flag = 0;
			MPI_Iprobe(MPI_ANY_SOURCE, TAG_REQ_WORKER, MPI_COMM_WORLD, &flag, &st);
			if (flag)
			{
				int req_depth;
				MPI_Recv(&req_depth, 1, MPI_INT, st.MPI_SOURCE, TAG_REQ_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				int assigned = -1;
				if (!free_workers.empty())
				{
					assigned = free_workers.front();
					free_workers.pop();
				}
				MPI_Send(&assigned, 1, MPI_INT, st.MPI_SOURCE, TAG_ASSIGN, MPI_COMM_WORLD);
				continue;
			}

			MPI_Iprobe(MPI_ANY_SOURCE, TAG_WORKER_FREE, MPI_COMM_WORLD, &flag, &st);
			if (flag)
			{
				int wid;
				MPI_Recv(&wid, 1, MPI_INT, st.MPI_SOURCE, TAG_WORKER_FREE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				free_workers.push(wid);
				continue;
			}

			std::this_thread::sleep_for(std::chrono::milliseconds(1));

		}

	}

	static void Strassen_distributed(SplittableMatrix<T>& C, SplittableMatrix<T>& A, SplittableMatrix<T>& B, int depth)
	{
		int w = request_worker(depth);
		if (w == -1)
		{
			#pragma omp parallel
			{
				#pragma omp single
				{
					StrassenRecursive(C, A, B, depth + 1);
				}
			}
			// StrassenRecursive(C, A, B, depth + 1);
			return;
		}
		else 
		{
			int out_id = -1;
			send_data(A, B, A.cdim, depth, w, C);
		}
	}

		
	static void StrassenRecursive(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B, int depth) 
	{
        // const size_t N = std::max(C.rdim, C.cdim);
		const size_t N = C.cdim;
        if (N <= threshold) 
		{
            Seq<T>::operate(C, A, B);
            return;
        }

		size_t hsize = N / 2;
		size_t rem_size = N - hsize;

        // split A, B, C into 4 submatrices using SplittableMatrix::split or view()
		SplittableMatrix<T>* A00 = A.split(0, 0);
		SplittableMatrix<T>* A01 = A.split(0, 1);
		SplittableMatrix<T>* A10 = A.split(1, 0);
		SplittableMatrix<T>* A11 = A.split(1, 1);
		SplittableMatrix<T>* B00 = B.split(0, 0);
		SplittableMatrix<T>* B01 = B.split(0, 1);
		SplittableMatrix<T>* B10 = B.split(1, 0);
		SplittableMatrix<T>* B11 = B.split(1, 1);
			

		SplittableMatrix<T>* S0 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T>* S1 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T>* S2 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T>* S3 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T>* S4 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T>* S5 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T>* S6 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T>* S7 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T>* S8 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T>* S9 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);

		#pragma omp taskgroup
		{
			#pragma omp task shared(S0) private(A00, A11)
			{
				ufunc::addition::OmpVanilla<T>::operate(*S0, *A00, *A11); // S0 = A00 + A11
			}
			#pragma omp task shared(S1) private(B00, B11)
			{
				ufunc::addition::OmpVanilla<T>::operate(*S1, *B00, *B11); // S1 = B00 + B11
			}
			#pragma omp task shared(S2) private(A10, A11)
			{
				ufunc::addition::OmpVanilla<T>::operate(*S2, *A10, *A11); 
			}
			#pragma omp task shared(S3) private(B01, B11)
			{
				ufunc::addition::OmpVanilla<T>::re_operate(*S3, *B01, *B11); 
			}
			#pragma omp task shared(S4) private(B10, B00)
			{
				ufunc::addition::OmpVanilla<T>::re_operate(*S4, *B10, *B00); 
			}
			#pragma omp task shared(S5) private(A00, A01)
			{
				ufunc::addition::OmpVanilla<T>::operate(*S5, *A00, *A01); 
			}
			#pragma omp task shared(S6) private(A10, A00)
			{
				ufunc::addition::OmpVanilla<T>::re_operate(*S6, *A10, *A00); 
			}
			#pragma omp task shared(S7) private(B00, B01)
			{
				ufunc::addition::OmpVanilla<T>::operate(*S7, *B00, *B01);
			}
			#pragma omp task shared(S8) private(A01, A11)
			{
				ufunc::addition::OmpVanilla<T>::re_operate(*S8, *A01, *A11);
			}
			#pragma omp task shared(S9) private(B10, B11)
			{
				ufunc::addition::OmpVanilla<T>::operate(*S9, *B10, *B11); 
			}
		}
		SplittableMatrix<T>* M[7];
		for (int i = 0; i < 7; i++) 
		{
			M[i] = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		}
		
		Strassen_distributed(*M[0], *S0, *S1, depth +1);

		Strassen_distributed(*M[1], *S2, *B00, depth +1);

		Strassen_distributed(*M[2], *A00, *S3, depth +1);

		Strassen_distributed(*M[3], *A11, *S4, depth +1);

		Strassen_distributed(*M[4], *S5, *B11, depth +1);

		Strassen_distributed(*M[5], *S6, *S7, depth +1);

		Strassen_distributed(*M[6], *S8, *S9, depth +1);

		delete A00; delete A01; delete A10; delete A11;
		delete B00; delete B01; delete B10; delete B11;
		delete S0; delete S1; delete S2; delete S3; delete S4;
		delete S5; delete S6; delete S7; delete S8; delete S9;
		
		
		#pragma omp taskgroup
		{
			#pragma omp task
			{
				ufunc::addition::Seq<T>::operate(*M[6], *M[6], *M[0]);
		
				ufunc::addition::Seq<T>::operate(*M[6], *M[6], *M[3]);

				ufunc::addition::Seq<T>::re_operate(*C.split(0, 0), *M[6], *M[4]);
			}
			#pragma omp task
			{
				ufunc::addition::Seq<T>::operate(*C.split(0, 1), *M[2], *M[4]);
			}
			#pragma omp task
			{
				ufunc::addition::Seq<T>::operate(*C.split(1, 0), *M[1], *M[3]);
			}
			#pragma omp task
			{
				ufunc::addition::Seq<T>::operate(*M[5], *M[5], *M[0]);

				ufunc::addition::Seq<T>::re_operate(*M[5], *M[5], *M[1]);

				ufunc::addition::Seq<T>::operate(*C.split(1, 1), *M[5], *M[2]);
			}
		}

		// Clean up
		

		for (int i = 0; i < 7; i++) {
			delete M[i];
		}
	}
};
template <typename T>
std::atomic<bool> HybridStrassen<T>::g_stop{false};

template <typename T>
size_t HybridStrassen<T>::threshold = -1;

template<class T>
class MPIStrassen {
public:
	static inline void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) 
	{
		int rank, nprocs; 
    	MPI_Comm_rank(MPI_COMM_WORLD,&rank); 
    	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0)
		{

			std::thread scheduler_thread(scheduler_loop, nprocs );

			StrassenRecursive(out, lhs,  rhs, 0);

			for (int i = 1; i < nprocs; ++i) 
				MPI_Send(NULL,0,MPI_BYTE,i,TAG_STOP,MPI_COMM_WORLD);
			g_stop.store(true);

			scheduler_thread.join();

		}
		else 
		{
			worker_loop(rank);
		}
	}

	static void set_threshold(size_t _threshold) 
	{
		threshold = _threshold;
	}

protected:
	static size_t threshold;
	SplittableMatrix<T>* out;
	const SplittableMatrix<T>* lhs;
	const SplittableMatrix<T>* rhs;

	static std::atomic<bool> g_stop;

	enum {
		TAG_TASK = 100, TAG_RESULT = 101, TAG_STOP = 102,
		TAG_REQ_WORKER = 103, TAG_ASSIGN = 104, TAG_WORKER_FREE = 105
	};

	struct task
	{
		int size;
		int caller;
		int depth;
	};

	MPIStrassen(SplittableMatrix<T>* out, const SplittableMatrix<T>* lhs, const SplittableMatrix<T>* rhs)
		: out(out), lhs(lhs), rhs(rhs) {}

	static void worker_loop(int myRank)
	{
		while (true)
		{
			int flag = 0;
			MPI_Status st;
			MPI_Iprobe(0, TAG_STOP, MPI_COMM_WORLD, &flag, &st);
			if (flag)
			{
				MPI_Recv(NULL, 0, MPI_BYTE, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				break;
			}
			
			MPI_Iprobe(MPI_ANY_SOURCE, TAG_TASK, MPI_COMM_WORLD, &flag, &st);
			if (!flag)
			{
				continue; 
			}

			task h;
			MPI_Recv(&h, 3, MPI_INT, st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			int n = h.size;

			SplittableMatrix<T> *A = new SplittableMatrix<T>(new Buffer<T>(n, n), n, n);
			SplittableMatrix<T> *B = new SplittableMatrix<T>(new Buffer<T>(n, n), n, n);

			MPI_Recv(A->root->data, n*n, sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(B->root->data, n*n, sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			StrassenRecursive(*A, *A, *B, h.depth);

			MPI_Send(&h, 3, MPI_INT, h.caller, TAG_RESULT, MPI_COMM_WORLD);
			MPI_Send(A->root->data, n*n, sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, h.caller, TAG_RESULT, MPI_COMM_WORLD);
			MPI_Send(&myRank, 1, MPI_INT, 0, TAG_WORKER_FREE, MPI_COMM_WORLD);

		}
	}

	static int request_worker(int depth) 
	{
		MPI_Send(&depth, 1, MPI_INT, 0, TAG_REQ_WORKER, MPI_COMM_WORLD);
		int worker; 
		MPI_Recv(&worker, 1, MPI_INT, 0, TAG_ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		return worker; 
	}

	static void send_data(SplittableMatrix<T>& a, SplittableMatrix<T>& b, int n, int depth, int worker, SplittableMatrix<T>& res)
	{
		task h {
			.size = n,
			.depth = depth
		};
		MPI_Comm_rank(MPI_COMM_WORLD, &h.caller);

		MPI_Send(&h, 3, MPI_INT, worker, TAG_TASK, MPI_COMM_WORLD);

		MPI_Send(a.root->data, n*n, sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, worker, TAG_TASK, MPI_COMM_WORLD);
		MPI_Send(b.root->data, n*n, sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, worker, TAG_TASK, MPI_COMM_WORLD);
		
		// nhận kết quả
		task rh;
		MPI_Recv(&rh, 3, MPI_INT, worker, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(res.root->data, rh.size*rh.size, sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, worker, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
	}

	static inline void scheduler_loop(int nprocs)
	{
		std::queue<int> free_workers;
		for (int w = 1; w < nprocs; ++w) 
			free_workers.push(w);

		MPI_Status st;

		while (!g_stop)
		{
			int flag = 0;
			MPI_Iprobe(MPI_ANY_SOURCE, TAG_REQ_WORKER, MPI_COMM_WORLD, &flag, &st);
			if (flag)
			{
				int req_depth;
				MPI_Recv(&req_depth, 1, MPI_INT, st.MPI_SOURCE, TAG_REQ_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				int assigned = -1;
				if (!free_workers.empty())
				{
					assigned = free_workers.front();
					free_workers.pop();
				}
				MPI_Send(&assigned, 1, MPI_INT, st.MPI_SOURCE, TAG_ASSIGN, MPI_COMM_WORLD);
				continue;
			}

			MPI_Iprobe(MPI_ANY_SOURCE, TAG_WORKER_FREE, MPI_COMM_WORLD, &flag, &st);
			if (flag)
			{
				int wid;
				MPI_Recv(&wid, 1, MPI_INT, st.MPI_SOURCE, TAG_WORKER_FREE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				free_workers.push(wid);
				continue;
			}

			std::this_thread::sleep_for(std::chrono::milliseconds(1));

		}

	}

	static void Strassen_distributed(SplittableMatrix<T>& C, SplittableMatrix<T>& A, SplittableMatrix<T>& B, int depth)
	{
		int w = request_worker(depth);
		if (w == -1)
		{
			StrassenRecursive(C, A, B, depth + 1);
			return;
		}
		else 
		{
			int out_id = -1;
			send_data(A, B, A.cdim, depth, w, C);
		}
	}

		
	static void StrassenRecursive(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B, int depth) 
	{
        // const size_t N = std::max(C.rdim, C.cdim);
		const size_t N = C.cdim;
        if (N <= threshold) 
		{
			CODE_FOR_DEBUG_MODE(printf("Rank %d: depth %d, size %zu handled locally\n", get_mpi_rank(), depth, N);)
            Seq<T>::operate(C, A, B);
            return;
        }

		size_t hsize = N / 2;
		size_t rem_size = N - hsize;

        // split A, B, C into 4 submatrices using SplittableMatrix::split or view()
        SplittableMatrix<T>* A00 = A.split(0, 0);
		SplittableMatrix<T>* A01 = A.split(0, 1);
		SplittableMatrix<T>* A10 = A.split(1, 0);
		SplittableMatrix<T>* A11 = A.split(1, 1);
		SplittableMatrix<T>* B00 = B.split(0, 0);
		SplittableMatrix<T>* B01 = B.split(0, 1);
		SplittableMatrix<T>* B10 = B.split(1, 0);
		SplittableMatrix<T>* B11 = B.split(1, 1);

		SplittableMatrix<T>* S0 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::operate(*S0, *A00, *A11); // S0 = A00 + A11
		
		SplittableMatrix<T>* S1 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::operate(*S1, *B00, *B11); // S1 = B00 + B11
		
		SplittableMatrix<T>* S2 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::operate(*S2, *A10, *A11); 

		SplittableMatrix<T>* S3 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::re_operate(*S3, *B01, *B11); 

		SplittableMatrix<T>* S4 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::re_operate(*S4, *B10, *B00); 

		SplittableMatrix<T>* S5 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::operate(*S5, *A00, *A01); 

		SplittableMatrix<T>* S6 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::re_operate(*S6, *A10, *A00); 

		SplittableMatrix<T>* S7 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::operate(*S7, *B00, *B01);
		
		SplittableMatrix<T>* S8 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::re_operate(*S8, *A01, *A11);
		
		SplittableMatrix<T>* S9 = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::operate(*S9, *B10, *B11); 

		SplittableMatrix<T>* M[7];
		for (int i = 0; i < 7; i++) 
		{
			M[i] = new SplittableMatrix<T>(new Buffer<T>(hsize), hsize);
		}
		
		Strassen_distributed(*M[0], *S0, *S1, depth +1);

		Strassen_distributed(*M[1], *S2, *B00, depth +1);

		Strassen_distributed(*M[2], *A00, *S3, depth +1);

		Strassen_distributed(*M[3], *A11, *S4, depth +1);

		Strassen_distributed(*M[4], *S5, *B11, depth +1);

		Strassen_distributed(*M[5], *S6, *S7, depth +1);

		Strassen_distributed(*M[6], *S8, *S9, depth +1);

		delete A00; delete A01; delete A10; delete A11;
		delete B00; delete B01; delete B10; delete B11;
		delete S0; delete S1; delete S2; delete S3; delete S4;
		delete S5; delete S6; delete S7; delete S8; delete S9;
		
		
		ufunc::addition::Seq<T>::operate(*M[6], *M[6], *M[0]);
		
		ufunc::addition::Seq<T>::operate(*M[6], *M[6], *M[3]);

		ufunc::addition::Seq<T>::re_operate(*C.split(0, 0), *M[6], *M[4]);

		ufunc::addition::Seq<T>::operate(*C.split(0, 1), *M[2], *M[4]);

		ufunc::addition::Seq<T>::operate(*C.split(1, 0), *M[1], *M[3]);

		ufunc::addition::Seq<T>::operate(*M[5], *M[5], *M[0]);

		ufunc::addition::Seq<T>::re_operate(*M[5], *M[5], *M[1]);

		ufunc::addition::Seq<T>::operate(*C.split(1, 1), *M[5], *M[2]);

		// Clean up
		

		for (int i = 0; i < 7; i++) {
			delete M[i];
		}
	}
};
template <typename T>
std::atomic<bool> MPIStrassen<T>::g_stop{false};

template <typename T>
size_t MPIStrassen<T>::threshold = -1;

#endif // USE_MPI


}; // namespace matmul
}; // namespace ufunc

// #endif // MATRIX_UFUNC_MATMUL_H
