#ifndef MATRIX_UFUNC_MATMUL_H
#define MATRIX_UFUNC_MATMUL_H

#include "../viewer.h"
#include "add.h"
#include <assert.h>

#ifdef MPI_ENABLE
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


// template<class T>
// class OmpVanilla
// {
// public:
// 	static inline void operate(Buffer<T>& out, const Buffer<T>& lhs, const Buffer<T>& rhs) {
// 		#pragma omp parallel for
// 		for (size_t i = 0; i < out.rdim; i++) {
// 			for (size_t k = 0; k < lhs.cdim; k++) {
// 				const T lhs_val = lhs.data[i * lhs.cdim + k];
// 				for (size_t j = 0; j < out.rdim; j++) {
// 					const T rhs_val = rhs.data[k * rhs.cdim + j];
// 					out[i * out.cdim + j] += lhs_val * rhs_val;
// 				}
// 			}
// 		}
// 	}

// 	static inline void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) {
// 		#pragma omp parallel for
// 		for (size_t i = 0; i < out.rdim; i++) {
// 			for template<class T>
// class HybridGrid {
// public:
// 	static const int THRESHOLD = 64;

// 	static void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) {
// 		int rank, size;
// 		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// 		MPI_Comm_size(MPI_COMM_WORLD, &size);

// 		int grid_size = static_cast<int>(std::sqrt(size));
// 		if (grid_size * grid_size != size) {
// 			if (rank == 0) {
// 				fprintf(stderr, "Error: Number of processes must be a perfect square\n");
// 			}
// 			MPI_Abort(MPI_COMM_WORLD, 1);
// 		}

// 		int n = static_cast<int>(out.rdim);
		
// 		if (rank == 0) {
// 			rootProcess(out, lhs, rhs, n, grid_size);
// 		} else {
// 			workerProcess(n, grid_size, rank);
// 		}
// 	}

// protected:
// 	// Hybrid recursive multiply: combines OpenMP task parallelism with recursive divide-and-conquer
// 	static void multiplyRecursive(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, 
// 	                              const SplittableMatrix<T>& B) {
// 		int m = static_cast<int>(C.rdim);
// 		int k = static_cast<int>(A.cdim);
// 		int n = static_cast<int>(C.cdim);
		
// 		if (m <= THRESHOLD && k <= THRESHOLD && n <= THRESHOLD) {
// 			// Base case: direct multiplication with OpenMP parallel for
// 			#pragma omp parallel for
// 			for (int i = 0; i < m; ++i) {
// 				for (int kk = 0; kk < k; ++kk) {
// 					size_t a_idx = A.map2Dto1DIndex(i, kk);
// 					T aik = A.get(a_idx);
// 					for (int j = 0; j < n; ++j) {
// 						size_t b_idx = B.map2Dto1DIndex(kk, j);
// 						size_t c_idx = C.map2Dto1DIndex(i, j);
// 						C.set(c_idx, C.get(c_idx) + aik * B.get(b_idx));
// 					}
// 				}
// 			}
// 			return;
// 		}

// 		// Recursive case: divide and conquer with OpenMP tasks
// 		int half_m = m / 2;
// 		int half_k = k / 2;
// 		int half_n = n / 2;
// 		int rem_m = m - half_m;
// 		int rem_k = k - half_k;
// 		int rem_n = n - half_n;

// 		// Create all submatrix views
// 		SplittableMatrix<T>* A00 = A.view(0, 0, half_m, half_k);
// 		SplittableMatrix<T>* A01 = A.view(0, half_k, half_m, rem_k);
// 		SplittableMatrix<T>* A10 = A.view(half_m, 0, rem_m, half_k);
// 		SplittableMatrix<T>* A11 = A.view(half_m, half_k, rem_m, rem_k);

// 		SplittableMatrix<T>* B00 = B.view(0, 0, half_k, half_n);
// 		SplittableMatrix<T>* B01 = B.view(0, half_n, half_k, rem_n);
// 		SplittableMatrix<T>* B10 = B.view(half_k, 0, rem_k, half_n);
// 		SplittableMatrix<T>* B11 = B.view(half_k, half_n, rem_k, rem_n);

// 		SplittableMatrix<T>* C00 = C.view(0, 0, half_m, half_n);
// 		SplittableMatrix<T>* C01 = C.view(0, half_n, half_m, rem_n);
// 		SplittableMatrix<T>* C10 = C.view(half_m, 0, rem_m, half_n);
// 		SplittableMatrix<T>* C11 = C.view(half_m, half_n, rem_m, rem_n);

// 		#pragma omp parallel
// 		{
// 			#pragma omp single
// 			{
// 				// C00 = A00*B00 + A01*B10
// 				#pragma omp task shared(C00, A00, B00)
// 				{
// 					multiplyRecursive(*C00, *A00, *B00);
// 				}
// 				#pragma omp task shared(C00, A01, B10)
// 				{
// 					multiplyRecursive(*C00, *A01, *B10);
// 				}

// 				// C01 = A00*B01 + A01*B11
// 				#pragma omp task shared(C01, A00, B01)
// 				{
// 					multiplyRecursive(*C01, *A00, *B01);
// 				}
// 				#pragma omp task shared(C01, A01, B11)
// 				{
// 					multiplyRecursive(*C01, *A01, *B11);
// 				}

// 				// C10 = A10*B00 + A11*B10
// 				#pragma omp task shared(C10, A10, B00)
// 				{
// 					multiplyRecursive(*C10, *A10, *B00);
// 				}
// 				#pragma omp task shared(C10, A11, B10)
// 				{
// 					multiplyRecursive(*C10, *A11, *B10);
// 				}

// 				// C11 = A10*B01 + A11*B11
// 				#pragma omp task shared(C11, A10, B01)
// 				{
// 					multiplyRecursive(*C11, *A10, *B01);
// 				}
// 				#pragma omp task shared(C11, A11, B11)
// 				{
// 					multiplyRecursive(*C11, *A11, *B11);
// 				}

// 				#pragma omp taskwait
// 			}
// 		}

// 		// Clean up views
// 		delete A00; delete A01; delete A10; delete A11;
// 		delete B00; delete B01; delete B10; delete B11;
// 		delete C00; delete C01; delete C10; delete C11;
// 	}

// 	static void rootProcess(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, 
// 	                       const SplittableMatrix<T>& B, int n, int grid_size) {
// 		int base_block_rows = n / grid_size;
// 		int base_block_cols = n / grid_size;
// 		int extra_rows = n % grid_size;
// 		int extra_cols = n % grid_size;

// 		MPI_Datatype mpi_type = sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE;

// 		// Create views for all blocks
// 		std::vector<std::vector<SplittableMatrix<T>*>> A_views(grid_size, 
// 			std::vector<SplittableMatrix<T>*>(grid_size));
// 		std::vector<std::vector<SplittableMatrix<T>*>> B_views(grid_size, 
// 			std::vector<SplittableMatrix<T>*>(grid_size));

// 		for (int grid_row = 0; grid_row < grid_size; ++grid_row) {
// 			int num_rows = base_block_rows + (grid_row < extra_rows ? 1 : 0);
// 			int row_start = grid_row * base_block_rows + std::min(grid_row, extra_rows);

// 			for (int grid_col = 0; grid_col < grid_size; ++grid_col) {
// 				int num_cols = base_block_cols + (grid_col < extra_cols ? 1 : 0);
// 				int col_start = grid_col * base_block_cols + std::min(grid_col, extra_cols);

// 				A_views[grid_row][grid_col] = A.view(row_start, col_start, num_rows, num_cols);
// 				B_views[grid_row][grid_col] = B.view(row_start, col_start, num_rows, num_cols);
// 			}
// 		}

// 		// Send blocks to workers using MPI derived datatypes
// 		std::vector<MPI_Request> send_requests;
// 		std::vector<MPI_Datatype> datatypes;

// 		for (int dest_rank = 1; dest_rank < grid_size * grid_size; ++dest_rank) {
// 			int dest_row = dest_rank / grid_size;
// 			int dest_col = dest_rank % grid_size;

// 			for (int k_block = 0; k_block < grid_size; ++k_block) {
// 				SplittableMatrix<T>* A_block = A_views[dest_row][k_block];
// 				SplittableMatrix<T>* B_block = B_views[k_block][dest_col];

// 				// Create MPI subarray datatype for A block
// 				MPI_Datatype A_subarray;
// 				int A_sizes[2] = {(int)A.root->rdim, (int)A.root->cdim};
// 				int A_subsizes[2] = {(int)A_block->rdim, (int)A_block->cdim};
// 				int A_starts[2] = {(int)A_block->rDis, (int)A_block->cDis};
// 				MPI_Type_create_subarray(2, A_sizes, A_subsizes, A_starts, 
// 				                         MPI_ORDER_C, mpi_type, &A_subarray);
// 				MPI_Type_commit(&A_subarray);
// 				datatypes.push_back(A_subarray);

// 				send_requests.push_back(MPI_Request());
// 				MPI_Isend(A.root->data, 1, A_subarray, dest_rank, 
// 				         k_block * 2, MPI_COMM_WORLD, &send_requests.back());

// 				// Create MPI subarray datatype for B block
// 				MPI_Datatype B_subarray;
// 				int B_sizes[2] = {(int)B.root->rdim, (int)B.root->cdim};
// 				int B_subsizes[2] = {(int)B_block->rdim, (int)B_block->cdim};
// 				int B_starts[2] = {(int)B_block->rDis, (int)B_block->cDis};
// 				MPI_Type_create_subarray(2, B_sizes, B_subsizes, B_starts, 
// 				                         MPI_ORDER_C, mpi_type, &B_subarray);
// 				MPI_Type_commit(&B_subarray);
// 				datatypes.push_back(B_subarray);

// 				send_requests.push_back(MPI_Request());
// 				MPI_Isend(B.root->data, 1, B_subarray, dest_rank, 
// 				         k_block * 2 + 1, MPI_COMM_WORLD, &send_requests.back());
// 			}
// 		}

// 		// Compute local result (rank 0's contribution) with OpenMP
// 		int local_rows = base_block_rows + (0 < extra_rows ? 1 : 0);
// 		int local_cols = base_block_cols + (0 < extra_cols ? 1 : 0);
		
// 		SplittableMatrix<T>* C_local = C.view(0, 0, local_rows, local_cols);

// 		for (int k_block = 0; k_block < grid_size; ++k_block) {
// 			SplittableMatrix<T>* A_local = A_views[0][k_block];
// 			SplittableMatrix<T>* B_local = B_views[k_block][0];
			
// 			// Use hybrid recursive multiply with OpenMP
// 			multiplyRecursive(*C_local, *A_local, *B_local);
// 		}
		
// 		delete C_local;

// 		// Receive results from workers
// 		for (int src_rank = 1; src_rank < grid_size * grid_size; ++src_rank) {
// 			int src_row = src_rank / grid_size;
// 			int src_col = src_rank % grid_size;
// 			int result_rows = base_block_rows + (src_row < extra_rows ? 1 : 0);
// 			int result_cols = base_block_cols + (src_col < extra_cols ? 1 : 0);
// 			int row_start = src_row * base_block_rows + std::min(src_row, extra_rows);
// 			int col_start = src_col * base_block_cols + std::min(src_col, extra_cols);

// 			std::vector<T> result_block(result_rows * result_cols);
// 			MPI_Recv(result_block.data(), result_rows * result_cols,
// 			        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, src_rank, 0,
// 			        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// 			SplittableMatrix<T>* C_result = C.view(row_start, col_start, result_rows, result_cols);
// 			for (int i = 0; i < result_rows; ++i) {
// 				for (int j = 0; j < result_cols; ++j) {
// 					C_result->set(C_result->map2Dto1DIndex(i, j), result_block[i * result_cols + j]);
// 				}
// 			}
// 			delete C_result;
// 		}

// 		MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);

// 		// Cleanup
// 		for (auto& dtype : datatypes) {
// 			MPI_Type_free(&dtype);
// 		}

// 		for (int i = 0; i < grid_size; ++i) {
// 			for (int j = 0; j < grid_size; ++j) {
// 				delete A_views[i][j];
// 				delete B_views[i][j];
// 			}
// 		}
// 	}

// 	static void workerProcess(int n, int grid_size, int rank) {
// 		int my_row = rank / grid_size;
// 		int my_col = rank % grid_size;

// 		int base_block_rows = n / grid_size;
// 		int base_block_cols = n / grid_size;
// 		int extra_rows = n % grid_size;
// 		int extra_cols = n % grid_size;

// 		int my_rows = base_block_rows + (my_row < extra_rows ? 1 : 0);
// 		int my_cols = base_block_cols + (my_col < extra_cols ? 1 : 0);

// 		// Create result buffer
// 		Buffer<T>* result_buffer = new Buffer<T>(my_rows, my_cols);
// 		for (int i = 0; i < my_rows * my_cols; ++i) {
// 			result_buffer->data[i] = static_cast<T>(0);
// 		}
// 		SplittableMatrix<T> result(result_buffer, my_rows, my_cols);

// 		for (int k_block = 0; k_block < grid_size; ++k_block) {
// 			int k_size = base_block_cols + (k_block < extra_cols ? 1 : 0);
// 			int a_rows = my_rows;
// 			int a_cols = k_size;
// 			int b_rows = k_size;
// 			int b_cols = my_cols;

// 			// Receive A and B blocks
// 			Buffer<T>* A_buffer = new Buffer<T>(a_rows, a_cols);
// 			Buffer<T>* B_buffer = new Buffer<T>(b_rows, b_cols);

// 			MPI_Recv(A_buffer->data, a_rows * a_cols, 
// 			        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, 0, k_block * 2,
// 			        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// 			MPI_Recv(B_buffer->data, b_rows * b_cols,
// 			        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, 0, k_block * 2 + 1,
// 			        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// 			// Wrap in SplittableMatrix
// 			SplittableMatrix<T> A_mat(A_buffer, a_rows, a_cols);
// 			SplittableMatrix<T> B_mat(B_buffer, b_rows, b_cols);

// 			// Compute with OpenMP hybrid recursion
// 			multiplyRecursive(result, A_mat, B_mat);

// 			delete A_buffer;
// 			delete B_buffer;
// 		}
// template<class T>
// class HybridGrid {
// public:
// 	static const int THRESHOLD = 64;

// 	static void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) {
// 		int rank, size;
// 		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// 		MPI_Comm_size(MPI_COMM_WORLD, &size);

// 		int grid_size = static_cast<int>(std::sqrt(size));
// 		if (grid_size * grid_size != size) {
// 			if (rank == 0) {
// 				fprintf(stderr, "Error: Number of processes must be a perfect square\n");
// 			}
// 			MPI_Abort(MPI_COMM_WORLD, 1);
// 		}

// 		int n = static_cast<int>(out.rdim);
		
// 		if (rank == 0) {
// 			rootProcess(out, lhs, rhs, n, grid_size);
// 		} else {
// 			workerProcess(n, grid_size, rank);
// 		}
// 	}

// protected:
// 	// Hybrid recursive multiply: combines OpenMP task parallelism with recursive divide-and-conquer
// 	static void multiplyRecursive(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, 
// 	                              const SplittableMatrix<T>& B) {
// 		int m = static_cast<int>(C.rdim);
// 		int k = static_cast<int>(A.cdim);
// 		int n = static_cast<int>(C.cdim);
		
// 		if (m <= THRESHOLD && k <= THRESHOLD && n <= THRESHOLD) {
// 			// Base case: direct multiplication with OpenMP parallel for
// 			#pragma omp parallel for
// 			for (int i = 0; i < m; ++i) {
// 				for (int kk = 0; kk < k; ++kk) {
// 					size_t a_idx = A.map2Dto1DIndex(i, kk);
// 					T aik = A.get(a_idx);
// 					for (int j = 0; j < n; ++j) {
// 						size_t b_idx = B.map2Dto1DIndex(kk, j);
// 						size_t c_idx = C.map2Dto1DIndex(i, j);
// 						C.set(c_idx, C.get(c_idx) + aik * B.get(b_idx));
// 					}
// 				}
// 			}
// 			return;
// 		}

// 		// Recursive case: divide and conquer with OpenMP tasks
// 		int half_m = m / 2;
// 		int half_k = k / 2;
// 		int half_n = n / 2;
// 		int rem_m = m - half_m;
// 		int rem_k = k - half_k;
// 		int rem_n = n - half_n;

// 		// Create all submatrix views
// 		SplittableMatrix<T>* A00 = A.view(0, 0, half_m, half_k);
// 		SplittableMatrix<T>* A01 = A.view(0, half_k, half_m, rem_k);
// 		SplittableMatrix<T>* A10 = A.view(half_m, 0, rem_m, half_k);
// 		SplittableMatrix<T>* A11 = A.view(half_m, half_k, rem_m, rem_k);

// 		SplittableMatrix<T>* B00 = B.view(0, 0, half_k, half_n);
// 		SplittableMatrix<T>* B01 = B.view(0, half_n, half_k, rem_n);
// 		SplittableMatrix<T>* B10 = B.view(half_k, 0, rem_k, half_n);
// 		SplittableMatrix<T>* B11 = B.view(half_k, half_n, rem_k, rem_n);

// 		SplittableMatrix<T>* C00 = C.view(0, 0, half_m, half_n);
// 		SplittableMatrix<T>* C01 = C.view(0, half_n, half_m, rem_n);
// 		SplittableMatrix<T>* C10 = C.view(half_m, 0, rem_m, half_n);
// 		SplittableMatrix<T>* C11 = C.view(half_m, half_n, rem_m, rem_n);

// 		#pragma omp parallel
// 		{
// 			#pragma omp single
// 			{
// 				// C00 = A00*B00 + A01*B10
// 				#pragma omp task shared(C00, A00, B00)
// 				{
// 					multiplyRecursive(*C00, *A00, *B00);
// 				}
// 				#pragma omp task shared(C00, A01, B10)
// 				{
// 					multiplyRecursive(*C00, *A01, *B10);
// 				}

// 				// C01 = A00*B01 + A01*B11
// 				#pragma omp task shared(C01, A00, B01)
// 				{
// 					multiplyRecursive(*C01, *A00, *B01);
// 				}
// 				#pragma omp task shared(C01, A01, B11)
// 				{
// 					multiplyRecursive(*C01, *A01, *B11);
// 				}

// 				// C10 = A10*B00 + A11*B10
// 				#pragma omp task shared(C10, A10, B00)
// 				{
// 					multiplyRecursive(*C10, *A10, *B00);
// 				}
// 				#pragma omp task shared(C10, A11, B10)
// 				{
// 					multiplyRecursive(*C10, *A11, *B10);
// 				}

// 				// C11 = A10*B01 + A11*B11
// 				#pragma omp task shared(C11, A10, B01)
// 				{
// 					multiplyRecursive(*C11, *A10, *B01);
// 				}
// 				#pragma omp task shared(C11, A11, B11)
// 				{
// 					multiplyRecursive(*C11, *A11, *B11);
// 				}

// 				#pragma omp taskwait
// 			}
// 		}

// 		// Clean up views
// 		delete A00; delete A01; delete A10; delete A11;
// 		delete B00; delete B01; delete B10; delete B11;
// 		delete C00; delete C01; delete C10; delete C11;
// 	}

// 	static void rootProcess(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, 
// 	                       const SplittableMatrix<T>& B, int n, int grid_size) {
// 		int base_block_rows = n / grid_size;
// 		int base_block_cols = n / grid_size;
// 		int extra_rows = n % grid_size;
// 		int extra_cols = n % grid_size;

// 		MPI_Datatype mpi_type = sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE;

// 		// Create views for all blocks
// 		std::vector<std::vector<SplittableMatrix<T>*>> A_views(grid_size, 
// 			std::vector<SplittableMatrix<T>*>(grid_size));
// 		std::vector<std::vector<SplittableMatrix<T>*>> B_views(grid_size, 
// 			std::vector<SplittableMatrix<T>*>(grid_size));

// 		for (int grid_row = 0; grid_row < grid_size; ++grid_row) {
// 			int num_rows = base_block_rows + (grid_row < extra_rows ? 1 : 0);
// 			int row_start = grid_row * base_block_rows + std::min(grid_row, extra_rows);

// 			for (int grid_col = 0; grid_col < grid_size; ++grid_col) {
// 				int num_cols = base_block_cols + (grid_col < extra_cols ? 1 : 0);
// 				int col_start = grid_col * base_block_cols + std::min(grid_col, extra_cols);

// 				A_views[grid_row][grid_col] = A.view(row_start, col_start, num_rows, num_cols);
// 				B_views[grid_row][grid_col] = B.view(row_start, col_start, num_rows, num_cols);
// 			}
// 		}

// 		// Send blocks to workers using MPI derived datatypes
// 		std::vector<MPI_Request> send_requests;
// 		std::vector<MPI_Datatype> datatypes;

// 		for (int dest_rank = 1; dest_rank < grid_size * grid_size; ++dest_rank) {
// 			int dest_row = dest_rank / grid_size;
// 			int dest_col = dest_rank % grid_size;

// 			for (int k_block = 0; k_block < grid_size; ++k_block) {
// 				SplittableMatrix<T>* A_block = A_views[dest_row][k_block];
// 				SplittableMatrix<T>* B_block = B_views[k_block][dest_col];

// 				// Create MPI subarray datatype for A block
// 				MPI_Datatype A_subarray;
// 				int A_sizes[2] = {(int)A.root->rdim, (int)A.root->cdim};
// 				int A_subsizes[2] = {(int)A_block->rdim, (int)A_block->cdim};
// 				int A_starts[2] = {(int)A_block->rDis, (int)A_block->cDis};
// 				MPI_Type_create_subarray(2, A_sizes, A_subsizes, A_starts, 
// 				                         MPI_ORDER_C, mpi_type, &A_subarray);
// 				MPI_Type_commit(&A_subarray);
// 				datatypes.push_back(A_subarray);

// 				send_requests.push_back(MPI_Request());
// 				MPI_Isend(A.root->data, 1, A_subarray, dest_rank, 
// 				         k_block * 2, MPI_COMM_WORLD, &send_requests.back());

// 				// Create MPI subarray datatype for B block
// 				MPI_Datatype B_subarray;
// 				int B_sizes[2] = {(int)B.root->rdim, (int)B.root->cdim};
// 				int B_subsizes[2] = {(int)B_block->rdim, (int)B_block->cdim};
// 				int B_starts[2] = {(int)B_block->rDis, (int)B_block->cDis};
// 				MPI_Type_create_subarray(2, B_sizes, B_subsizes, B_starts, 
// 				                         MPI_ORDER_C, mpi_type, &B_subarray);
// 				MPI_Type_commit(&B_subarray);
// 				datatypes.push_back(B_subarray);

// 				send_requests.push_back(MPI_Request());
// 				MPI_Isend(B.root->data, 1, B_subarray, dest_rank, 
// 				         k_block * 2 + 1, MPI_COMM_WORLD, &send_requests.back());
// 			}
// 		}

// 		// Compute local result (rank 0's contribution) with OpenMP
// 		int local_rows = base_block_rows + (0 < extra_rows ? 1 : 0);
// 		int local_cols = base_block_cols + (0 < extra_cols ? 1 : 0);
		
// 		SplittableMatrix<T>* C_local = C.view(0, 0, local_rows, local_cols);

// 		for (int k_block = 0; k_block < grid_size; ++k_block) {
// 			SplittableMatrix<T>* A_local = A_views[0][k_block];
// 			SplittableMatrix<T>* B_local = B_views[k_block][0];
			
// 			// Use hybrid recursive multiply with OpenMP
// 			multiplyRecursive(*C_local, *A_local, *B_local);
// 		}
		
// 		delete C_local;

// 		// Receive results from workers
// 		for (int src_rank = 1; src_rank < grid_size * grid_size; ++src_rank) {
// 			int src_row = src_rank / grid_size;
// 			int src_col = src_rank % grid_size;
// 			int result_rows = base_block_rows + (src_row < extra_rows ? 1 : 0);
// 			int result_cols = base_block_cols + (src_col < extra_cols ? 1 : 0);
// 			int row_start = src_row * base_block_rows + std::min(src_row, extra_rows);
// 			int col_start = src_col * base_block_cols + std::min(src_col, extra_cols);

// 			std::vector<T> result_block(result_rows * result_cols);
// 			MPI_Recv(result_block.data(), result_rows * result_cols,
// 			        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, src_rank, 0,
// 			        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// 			SplittableMatrix<T>* C_result = C.view(row_start, col_start, result_rows, result_cols);
// 			for (int i = 0; i < result_rows; ++i) {
// 				for (int j = 0; j < result_cols; ++j) {
// 					C_result->set(C_result->map2Dto1DIndex(i, j), result_block[i * result_cols + j]);
// 				}
// 			}
// 			delete C_result;
// 		}

// 		MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);

// 		// Cleanup
// 		for (auto& dtype : datatypes) {
// 			MPI_Type_free(&dtype);
// 		}

// 		for (int i = 0; i < grid_size; ++i) {
// 			for (int j = 0; j < grid_size; ++j) {
// 				delete A_views[i][j];
// 				delete B_views[i][j];
// 			}
// 		}
// 	}

// 	static void workerProcess(int n, int grid_size, int rank) {
// 		int my_row = rank / grid_size;
// 		int my_col = rank % grid_size;

// 		int base_block_rows = n / grid_size;
// 		int base_block_cols = n / grid_size;
// 		int extra_rows = n % grid_size;
// 		int extra_cols = n % grid_size;

// 		int my_rows = base_block_rows + (my_row < extra_rows ? 1 : 0);
// 		int my_cols = base_block_cols + (my_col < extra_cols ? 1 : 0);

// 		// Create result buffer
// 		Buffer<T>* result_buffer = new Buffer<T>(my_rows, my_cols);
// 		for (int i = 0; i < my_rows * my_cols; ++i) {
// 			result_buffer->data[i] = static_cast<T>(0);
// 		}
// 		SplittableMatrix<T> result(result_buffer, my_rows, my_cols);

// 		for (int k_block = 0; k_block < grid_size; ++k_block) {
// 			int k_size = base_block_cols + (k_block < extra_cols ? 1 : 0);
// 			int a_rows = my_rows;
// 			int a_cols = k_size;
// 			int b_rows = k_size;
// 			int b_cols = my_cols;

// 			// Receive A and B blocks
// 			Buffer<T>* A_buffer = new Buffer<T>(a_rows, a_cols);
// 			Buffer<T>* B_buffer = new Buffer<T>(b_rows, b_cols);

// 			MPI_Recv(A_buffer->data, a_rows * a_cols, 
// 			        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, 0, k_block * 2,
// 			        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// 			MPI_Recv(B_buffer->data, b_rows * b_cols,
// 			        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, 0, k_block * 2 + 1,
// 			        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// 			// Wrap in SplittableMatrix
// 			SplittableMatrix<T> A_mat(A_buffer, a_rows, a_cols);
// 			SplittableMatrix<T> B_mat(B_buffer, b_rows, b_cols);

// 			// Compute with OpenMP hybrid recursion
// 			multiplyRecursive(result, A_mat, B_mat);

// 			delete A_buffer;
// 			delete B_buffer;
// 		}

// 		// Send result back to root
// 		MPI_Send(result.root->data, my_rows * my_cols,
// 		        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

// 		delete result_buffer;
// 	}
// };
// 		// Send result back to root
// 		MPI_Send(result.root->data, my_rows * my_cols,
// 		        sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

// 		delete result_buffer;
// 	}
// };(size_t k = 0; k < lhs.cdim; k++) {
// 				const size_t lhs_idx = lhs.map2Dto1DIndex(i, k);
// 				const T lhs_val = lhs.get(lhs_idx);
// 				for (size_t j = 0; j < out.cdim; j++) {
// 					const size_t rhs_idx = rhs.map2Dto1DIndex(k, j);
// 					const size_t out_idx = out.map2Dto1DIndex(i, j);
// 					const T rhs_val = rhs.get(rhs_idx);
// 					const T prev_out = out.get(out_idx);
					
// 					out.set(out_idx, prev_out + lhs_val * rhs_val);
// 				}
// 			}
// 		}
// 	}
// };


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
class SeqStrassen {
public:
	static inline void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) 
	{
		StrassenRecursive(out, lhs, rhs);
		SplittableMatrix<T> ref_result(new Buffer<T>(out.rdim), out.rdim);
		ufunc::matmul::Seq<float>::operate(ref_result, lhs, rhs);
		CODE_FOR_DEBUG_MODE(                                      
                if (!SplittableMatrix<T>::is_same(out, ref_result)) {    
                    printf("[ERROR][BM_OMP_Strassen] Result mismatch at threshold \n" ); 
                }                                                          
                else                                                    
                    printf("[DEBUG][BM_OMP_Strassen] Result match at threshold \n" ); 
            )  

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
	SeqStrassen(SplittableMatrix<T>* out, const SplittableMatrix<T>* lhs, const SplittableMatrix<T>* rhs)
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

        // split A, B, C into 4 submatrices using SplittableMatrix::split or view()
		SplittableMatrix<T> M1(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M2(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M3(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> M4(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M5(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M6(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> M7(new Buffer<T>(hsize), hsize);

		SplittableMatrix<T>* A00 = A.split(0, 0);
		SplittableMatrix<T>* A01 = A.split(0, 1);
		SplittableMatrix<T>* A10 = A.split(1, 0);
		SplittableMatrix<T>* A11 = A.split(1, 1);
		SplittableMatrix<T>* B00 = B.split(0, 0);
		SplittableMatrix<T>* B01 = B.split(0, 1);
		SplittableMatrix<T>* B10 = B.split(1, 0);
		SplittableMatrix<T>* B11 = B.split(1, 1);
		SplittableMatrix<T>* C00 = C.split(0, 0);
		SplittableMatrix<T>* C01 = C.split(0, 1);
		SplittableMatrix<T>* C10 = C.split(1, 0);
		SplittableMatrix<T>* C11 = C.split(1, 1);

		SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::operate(Temp1, *A00, *A11);
		ufunc::addition::Seq<T>::operate(Temp2, *B00, *B11);
		StrassenRecursive(M1, Temp1, Temp2);

		// SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::operate(Temp1, *A10, *A11);
		StrassenRecursive(M2, Temp1, *B00);

		// SplittableMatrix<T> Temp4(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::re_operate(Temp1, *B01, *B11);
		StrassenRecursive(M3, *A00, Temp1);

		// SplittableMatrix<T> Temp5(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::re_operate(Temp1, *B10, *B00);
		StrassenRecursive(M4, *A11, Temp1);

		// SplittableMatrix<T> Temp6(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
		StrassenRecursive(M5, Temp1, *B11);

		// SplittableMatrix<T> Temp7(new Buffer<T>(hsize), hsize);
		// SplittableMatrix<T> Temp8(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::re_operate(Temp1, *A10, *A00);
		ufunc::addition::Seq<T>::operate(Temp2, *B00, *B01);
		StrassenRecursive(M6, Temp1, Temp2);

		// SplittableMatrix<T> Temp(new Buffer<T>(hsize), hsize);
		// SplittableMatrix<T> Temp10(new Buffer<T>(hsize), hsize);
		ufunc::addition::Seq<T>::re_operate(Temp1, *A01, *A11);
		ufunc::addition::Seq<T>::operate(Temp2, *B10, *B11);
		StrassenRecursive(M7, Temp1, Temp2);

		delete Temp1.root;
		delete Temp2.root;

		ufunc::addition::Seq<T>::operate(*C11, M3, M6);
		delete M6.root;
		
		ufunc::addition::Seq<T>::operate(*C00, M4, M7);
		delete M7.root;

		ufunc::addition::Seq<T>::operate(*C10, M2, M4);
		delete M4.root;
		
		ufunc::addition::Seq<T>::operate(*C01, M3, M5);
		delete M3.root;
		ufunc::addition::Seq<T>::re_operate(*C00, *C00, M5);
		delete M5.root;
	

		ufunc::addition::Seq<T>::re_operate(*C11, *C11, M2);
		delete M2.root;

		ufunc::addition::Seq<T>::operate(*C00, *C00, M1);
		ufunc::addition::Seq<T>::operate(*C11, *C11, M1);
		delete M1.root;

		delete A00;
		delete A01;
		delete A10;
		delete A11;
		delete B00;
		delete B01;
		delete B10;
		delete B11;

		delete C00;
		delete C01;
		delete C10;
		delete C11;

	}
};
template<class T> 
size_t SeqStrassen<T>::threshold = -1;


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
        // split A, B, C into 4 submatrices using SplittableMatrix::split or view()
		SplittableMatrix<T> M1(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M2(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M3(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> M4(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M5(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M6(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> M7(new Buffer<T>(hsize), hsize);

		SplittableMatrix<T>* A00 = A.split(0, 0);
		SplittableMatrix<T>* A01 = A.split(0, 1);
		SplittableMatrix<T>* A10 = A.split(1, 0);
		SplittableMatrix<T>* A11 = A.split(1, 1);
		SplittableMatrix<T>* B00 = B.split(0, 0);
		SplittableMatrix<T>* B01 = B.split(0, 1);
		SplittableMatrix<T>* B10 = B.split(1, 0);
		SplittableMatrix<T>* B11 = B.split(1, 1);
		SplittableMatrix<T>* C00 = C.split(0, 0);
		SplittableMatrix<T>* C01 = C.split(0, 1);
		SplittableMatrix<T>* C10 = C.split(1, 0);
		SplittableMatrix<T>* C11 = C.split(1, 1);

        #pragma omp taskgroup
		{
			#pragma omp task 
			{	
				SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
                SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
				ufunc::addition::Seq<T>::operate(Temp1, *A00, *A11);
				ufunc::addition::Seq<T>::operate(Temp2, *B00, *B11);
				StrassenRecursive(M1, Temp1, Temp2);
				#pragma omp taskwait
				delete Temp1.root;
				delete Temp2.root;
			}
			#pragma omp task 
			{
				SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
				ufunc::addition::Seq<T>::operate(Temp1, *A10, *A11);
				StrassenRecursive(M2, Temp1, *B00);
				#pragma omp taskwait
				delete Temp1.root;
			}
			#pragma omp task 
			{
				SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
				ufunc::addition::Seq<T>::re_operate(Temp1, *B01, *B11);
				StrassenRecursive(M3, *A00, Temp1);
				#pragma omp taskwait
				delete Temp1.root;
			}
			#pragma omp task 
			{
				// M4
				SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
				ufunc::addition::Seq<T>::re_operate(Temp1, *B10, *B00);
				StrassenRecursive(M4, *A11, Temp1);
				#pragma omp taskwait
				delete Temp1.root;

			}
			#pragma omp task 
			{
				SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);

				ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
				StrassenRecursive(M5, Temp1, *B11);
				#pragma omp taskwait
				delete Temp1.root;
			}
			#pragma omp task 
			{
				SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
				SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
				ufunc::addition::Seq<T>::re_operate(Temp1, *A10, *A00);
				ufunc::addition::Seq<T>::operate(Temp2, *B00, *B01);
				StrassenRecursive(M6, Temp1, Temp2);
				#pragma omp taskwait
				delete Temp1.root;
				delete Temp2.root;
			}
			#pragma omp task 
			{
				SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
				SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
				ufunc::addition::Seq<T>::re_operate(Temp1, *A01, *A11);
				ufunc::addition::Seq<T>::operate(Temp2, *B10, *B11);
				StrassenRecursive(M7, Temp1, Temp2);
				#pragma omp taskwait
				delete Temp1.root;
				delete Temp2.root;
			}
		}
		#pragma omp taskwait

		delete A00;
		delete A01;
		delete A10;
		delete A11;

		delete B00;
		delete B01;
		delete B10;
		delete B11;

		ufunc::addition::Seq<T>::operate(*C11, M3, M6);
		delete M6.root;
		
		ufunc::addition::Seq<T>::operate(*C00, M4, M7);
		delete M7.root;

		ufunc::addition::Seq<T>::operate(*C10, M2, M4);
		delete M4.root;
		
		ufunc::addition::Seq<T>::operate(*C01, M3, M5);
		delete M3.root;
		ufunc::addition::Seq<T>::re_operate(*C00, *C00, M5);
		delete M5.root;
	

		ufunc::addition::Seq<T>::re_operate(*C11, *C11, M2);
		delete M2.root;

		ufunc::addition::Seq<T>::operate(*C00, *C00, M1);
		ufunc::addition::Seq<T>::operate(*C11, *C11, M1);
		delete M1.root;

		delete C00;
		delete C01;
		delete C10;
		delete C11;
	}
};


template<class T> 
size_t OmpStrassen<T>::threshold = -1;

#ifdef MPI_ENABLE
// Baseline MPI implementation using simple row distribution and IKJ loop order
template<class T>
class HybridStrassen {
public:
	// MPI_Datatype mpi_type = sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE;

	// static std::atomic<bool> g_stop;
	static inline void operate(SplittableMatrix<T>& out,
                           const SplittableMatrix<T>& lhs,
                           const SplittableMatrix<T>& rhs)
	{
		int rank, nprocs;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

		// 1) Broadcast N (dimension) để mọi rank đồng nhất
		int N;
		if (rank == 0) N = (int)lhs.cdim;  // giả định lhs vuông
		MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// 2) Đảm bảo worker có buffer đúng size trước khi nhận data
		SplittableMatrix<T> A = lhs;
		SplittableMatrix<T> B = rhs;

		if (rank != 0) {
			A = SplittableMatrix<T>(new Buffer<T>(N), N);
			B = SplittableMatrix<T>(new Buffer<T>(N), N);

			// out cũng phải cùng size để lát Bcast kết quả
			out = SplittableMatrix<T>(new Buffer<T>(N), N);
		}

		// 3) Bcast data với count giống nhau trên mọi rank
		MPI_Bcast(A.root->data, N*N, mpi_type, 0, MPI_COMM_WORLD);
		MPI_Bcast(B.root->data, N*N, mpi_type, 0, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);

		// >>> từ đây trở đi dùng A,B thay vì lhs,rhs <<<
		if (rank == 0) {
			if ((size_t)N <= threshold) {
				Seq<T>::operate(out, A, B);
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(out.root->data, N*N, mpi_type, 0, MPI_COMM_WORLD);
				return;
			}
			StrassenRecursive(out, A, B);
		} else {
			if ((size_t)N <= threshold) {
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(out.root->data, N*N, mpi_type, 0, MPI_COMM_WORLD);
				return;
			}
			StrassenRecursive(out, A, B);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(out.root->data, N*N, mpi_type, 0, MPI_COMM_WORLD);
		// if (rank ==0)
		// {
		// 	SplittableMatrix<T> checkC(new Buffer<T>(N),N);
		// 	Seq<T>::operate(checkC,lhs,rhs);
		// 	SplittableMatrix<T>::is_same(checkC, out);
		// }
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
	static inline MPI_Datatype mpi_type = sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE;


	HybridStrassen(SplittableMatrix<T>* out, const SplittableMatrix<T>* lhs, const SplittableMatrix<T>* rhs)
		: out(out), lhs(lhs), rhs(rhs) {}


	static void StrassenRecursiveLocal(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B)
	{
		OmpStrassen<T>::operate(C, A, B);
	}

	static void StrassenRecursive_group2(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B, MPI_Comm subcomm, int i)
	{
		int sub_rank, sub_size;
		int world_rank;
		MPI_Comm_rank(subcomm, &sub_rank);
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		MPI_Comm_size(subcomm, &sub_size);

		const size_t N = C.cdim;
		if (N <= threshold) {
			Seq<T>::operate(C, A, B);
			return;
		}

		size_t h = N/2;

		// Allocate 7 blocks
		SplittableMatrix<T> M1(new Buffer<T>(h),h);
		SplittableMatrix<T> M2(new Buffer<T>(h),h);
		SplittableMatrix<T> M3(new Buffer<T>(h),h);
		SplittableMatrix<T> M4(new Buffer<T>(h),h);
		SplittableMatrix<T> M5(new Buffer<T>(h),h);
		SplittableMatrix<T> M6(new Buffer<T>(h),h);
		SplittableMatrix<T> M7(new Buffer<T>(h),h);

		SplittableMatrix<T>* A00=A.split(0,0);
		SplittableMatrix<T>* A01=A.split(0,1);
		SplittableMatrix<T>* A10=A.split(1,0);
		SplittableMatrix<T>* A11=A.split(1,1);
		SplittableMatrix<T>* B00=B.split(0,0);
		SplittableMatrix<T>* B01=B.split(0,1);
		SplittableMatrix<T>* B10=B.split(1,0);
		SplittableMatrix<T>* B11=B.split(1,1);

		SplittableMatrix<T> T1(new Buffer<T>(h),h);
		SplittableMatrix<T> T2(new Buffer<T>(h),h);

		if (sub_rank == 0)
		{
			// Compute M1-M4
			ufunc::addition::Seq<T>::operate(T1,*A00,*A11);
			ufunc::addition::Seq<T>::operate(T2,*B00,*B11);
			StrassenRecursiveLocal(M1,T1,T2);

			ufunc::addition::Seq<T>::operate(T1,*A10,*A11);
			StrassenRecursiveLocal(M2,T1,*B00);

			ufunc::addition::Seq<T>::re_operate(T1,*B01,*B11);
			StrassenRecursiveLocal(M3,*A00,T1);

			ufunc::addition::Seq<T>::re_operate(T2,*B10,*B00);
			StrassenRecursiveLocal(M4,*A11,T2);

			// Receive M5-M7 from sub_rank 1
			MPI_Recv(M5.root->data, h*h, mpi_type, 1, 14, subcomm, MPI_STATUS_IGNORE);
			MPI_Recv(M6.root->data, h*h, mpi_type, 1, 15, subcomm, MPI_STATUS_IGNORE);
			MPI_Recv(M7.root->data, h*h, mpi_type, 1, 16, subcomm, MPI_STATUS_IGNORE);

			SplittableMatrix<T>* C00=C.split(0,0);
			SplittableMatrix<T>* C01=C.split(0,1);
			SplittableMatrix<T>* C10=C.split(1,0); 
			SplittableMatrix<T>* C11=C.split(1,1);

			ufunc::addition::Seq<T>::operate(*C11, M3, M6);
			// delete M6.root;
			
			ufunc::addition::Seq<T>::operate(*C00, M4, M7);
			// delete M7.root;

			ufunc::addition::Seq<T>::operate(*C10, M2, M4);
			// delete M4.root;
			
			ufunc::addition::Seq<T>::operate(*C01, M3, M5);
			// delete M3.root;
			ufunc::addition::Seq<T>::re_operate(*C00, *C00, M5);
			// delete M5.root;
		
			ufunc::addition::Seq<T>::re_operate(*C11, *C11, M2);
			// delete M2.root;

			ufunc::addition::Seq<T>::operate(*C00, *C00, M1);
			ufunc::addition::Seq<T>::operate(*C11, *C11, M1);
			delete C00;
			delete C01;
			delete C10;
			delete C11;

		}
		else 
		{
			// Compute M5-M7
			ufunc::addition::Seq<T>::operate(T1,*A00,*A01);
			StrassenRecursiveLocal(M5,T1,*B11);

			ufunc::addition::Seq<T>::re_operate(T1,*A10,*A00);
			ufunc::addition::Seq<T>::operate(T2,*B00,*B01);
			StrassenRecursiveLocal(M6,T1,T2);

			ufunc::addition::Seq<T>::re_operate(T1,*A01,*A11);
			ufunc::addition::Seq<T>::operate(T2,*B10,*B11);
			StrassenRecursiveLocal(M7,T1,T2);
			// Send M5-M7 to sub_rank 0
			MPI_Send(M5.root->data,h*h,mpi_type,0,14,subcomm);
			MPI_Send(M6.root->data,h*h,mpi_type,0,15,subcomm);
			MPI_Send(M7.root->data,h*h,mpi_type,0,16,subcomm);
		}
		MPI_Request req = MPI_REQUEST_NULL;
		if (i == 1)
		{
			MPI_Bcast(C.root->data, N*N, mpi_type, 0, subcomm);
			// if (world_rank == 7)
			// {
			// 	MPI_Send(C.root->data, N*N, mpi_type, 0, 10, MPI_COMM_WORLD);
			// }
		}
		else if (i==2)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 11, MPI_COMM_WORLD, &req);
			}
		}
		else if (i == 3)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 12, MPI_COMM_WORLD, &req);
			}
		}
		else if (i==4)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 13, MPI_COMM_WORLD, &req);
			}
		}
		else if (i==5)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 14, MPI_COMM_WORLD, &req);
			}
		}
		else if (i==6)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 15, MPI_COMM_WORLD, &req);
			}
		}
		else if (i==7)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 16, MPI_COMM_WORLD, &req);
			}
		}
		MPI_Wait(&req, MPI_STATUS_IGNORE);
		delete M1.root;
		delete M2.root;
		delete M3.root;
		delete M4.root;
		delete M5.root;
		delete M6.root;
		delete M7.root;
		delete T1.root;
		delete T2.root;
		delete A00;
		delete A01;
		delete A10;
		delete A11;
		delete B00;
		delete B01;
		delete B10;
		delete B11;

	}

	static void StrassenRecursiveM1_2(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B)
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (A.cdim <= threshold)
		{
			if (rank == 0)
			{
				Seq<T>::operate(C, A, B);
			}
			return;
		}

		int color = (rank == 0 || rank == 7) ? 1 : MPI_UNDEFINED;
		MPI_Group world_group, sub_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		int ranks[2] = {0, 7};
		MPI_Group_incl(world_group, 2, ranks, &sub_group);

		MPI_Comm subcomm = MPI_COMM_NULL;

		MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &subcomm);

		if (subcomm != MPI_COMM_NULL) {
			StrassenRecursive_group2(C, A, B, subcomm, 1);
			MPI_Barrier(subcomm);
			MPI_Comm_free(&subcomm);
		}
		MPI_Group_free(&sub_group);
		MPI_Group_free(&world_group);
	}

		
	static void StrassenRecursive(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B) 
	{
        // const size_t N = std::max(C.rdim, C.cdim);
		const size_t N = C.cdim;
        if (N <= threshold) 
		{
			// CODE_FOR_DEBUG_MODE(printf("Rank: size %zu handled locally\n",  N);)
            Seq<T>::operate(C, A, B);
            return;
        }
		size_t hsize = N / 2;
		int rank = -1;
		int procs = -1;
		MPI_Comm_rank(MPI_COMM_WORLD,&rank); 
		MPI_Comm_size(MPI_COMM_WORLD,&procs);

        // split A, B, C into 4 submatrices using SplittableMatrix::split or view()
		SplittableMatrix<T> M1(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M2(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M3(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> M4(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M5(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M6(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> M7(new Buffer<T>(hsize), hsize);

		SplittableMatrix<T>* A00 = A.split(0, 0);
		SplittableMatrix<T>* A01 = A.split(0, 1);
		SplittableMatrix<T>* A10 = A.split(1, 0);
		SplittableMatrix<T>* A11 = A.split(1, 1);
		SplittableMatrix<T>* B00 = B.split(0, 0);
		SplittableMatrix<T>* B01 = B.split(0, 1);
		SplittableMatrix<T>* B10 = B.split(1, 0);
		SplittableMatrix<T>* B11 = B.split(1, 1);

		if (procs <= 7)
		{
			for (int i =0;i <7;i++)
			{
				int group = i%procs;
				if (rank == group)
				{
					switch (i)
					{
						case 0:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B11);
							StrassenRecursiveLocal(M1, Temp1, Temp2);
							// MPI_Send(M1.root->data, hsize*hsize, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 1:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A10, *A11);
							StrassenRecursiveLocal(M2, Temp1, *B00);
							// if (rank != 0)
							// 	MPI_Send(M2.root->data, hsize*hsize, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD);
							delete Temp1.root;
							break;
						}
						case 2:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *B01, *B11);
							StrassenRecursiveLocal(M3, *A00, Temp1);
								// if (rank != 0)
								// 	MPI_Send(M3.root->data, hsize*hsize, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
							// MPI_Send(M3.root->data, hsize*hsize, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
							delete Temp1.root;
							break;
						}
						case 3:
						{
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp2, *B10, *B00);
							StrassenRecursiveLocal(M4, *A11, Temp2);
							// if (rank != 0)
							// 	MPI_Send(M4.root->data, hsize*hsize, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD);
							delete Temp2.root;
							break;
						}
						case 4:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
							StrassenRecursiveLocal(M5, Temp1, *B11);
							// if (rank != 0)
							// 	MPI_Send(M5.root->data, hsize*hsize, MPI_DOUBLE, 0, 14, MPI_COMM_WORLD);
							delete Temp1.root;
							break;
						}
						case 5:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A10, *A00);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B01);
							StrassenRecursiveLocal(M6, Temp1, Temp2);
							// if (rank != 0)
							// 	MPI_Send(M6.root->data, hsize*hsize, MPI_DOUBLE, 0, 15, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 6:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A01, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B10, *B11);
							StrassenRecursiveLocal(M7, Temp1, Temp2);
							// if (rank != 0)
							// 	MPI_Send(M7.root->data, hsize*hsize, MPI_DOUBLE, 0, 16, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
					}
				}

			}
			if (procs == 2)
			{
				if (rank == 0)
				{
					MPI_Recv(M2.root->data, hsize*hsize, mpi_type, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M4.root->data, hsize*hsize, mpi_type, 1, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M6.root->data, hsize*hsize, mpi_type, 1, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				else if (rank ==1)
				{
					MPI_Send(M2.root->data, hsize*hsize, mpi_type, 0, 11, MPI_COMM_WORLD);
					MPI_Send(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD);
					MPI_Send(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD);

				}
			}
			else if (procs ==3)
			{
				if (rank == 0)
				{
					MPI_Recv(M2.root->data, hsize*hsize, mpi_type, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M3.root->data, hsize*hsize, mpi_type, 2, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					// MPI_Recv(M4.root->data, hsize*hsize, mpi_type, 1, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M5.root->data, hsize*hsize, mpi_type, 1, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M6.root->data, hsize*hsize, mpi_type, 2, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					// MPI_Recv(M7.root->data, hsize*hsize, mpi_type, 2, 16, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				if (rank ==1)
				{
					MPI_Send(M2.root->data, hsize*hsize, mpi_type, 0, 11, MPI_COMM_WORLD);
					// MPI_Send(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD);
					MPI_Send(M5.root->data, hsize*hsize, mpi_type, 0, 14, MPI_COMM_WORLD);

				}
				else if (rank ==2)
				{
					MPI_Send(M3.root->data, hsize*hsize, mpi_type, 0, 12, MPI_COMM_WORLD);
					MPI_Send(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
			}
			else if (procs == 4)
			{
				if (rank ==0)
				{
					MPI_Recv(M2.root->data, hsize*hsize, mpi_type, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M3.root->data, hsize*hsize, mpi_type, 2, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M4.root->data, hsize*hsize, mpi_type, 3, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					// MPI_Recv(M5.root->data, hsize*hsize, mpi_type, 1, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M6.root->data, hsize*hsize, mpi_type, 1, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M7.root->data, hsize*hsize, mpi_type, 2, 16, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				if (rank ==1)
				{
					MPI_Send(M2.root->data, hsize*hsize, mpi_type, 0, 11, MPI_COMM_WORLD);
					MPI_Send(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD);
				}
				else if (rank ==2)
				{
					MPI_Send(M3.root->data, hsize*hsize, mpi_type, 0, 12, MPI_COMM_WORLD);
					MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank ==3)
				{
					MPI_Send(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
			}
			else if (procs == 5)
			{
				if (rank ==0)
				{
					MPI_Recv(M2.root->data, hsize*hsize, mpi_type, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M3.root->data, hsize*hsize, mpi_type, 2, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M4.root->data, hsize*hsize, mpi_type, 3, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M5.root->data, hsize*hsize, mpi_type, 4, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					// MPI_Recv(M6.root->data, hsize*hsize, mpi_type, 2, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M7.root->data, hsize*hsize, mpi_type, 1, 16, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				if (rank ==1)
				{
					MPI_Send(M2.root->data, hsize*hsize, mpi_type, 0, 11, MPI_COMM_WORLD);
					MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank ==2)
				{
					MPI_Send(M3.root->data, hsize*hsize, mpi_type, 0, 12, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank ==3)
				{
					MPI_Send(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank == 4)
				{
					MPI_Send(M5.root->data, hsize*hsize, mpi_type, 0, 14, MPI_COMM_WORLD);
				}
		
			}
			else if (procs == 6)
			{
				if (rank ==0)
				{
					MPI_Recv(M2.root->data, hsize*hsize, mpi_type, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M3.root->data, hsize*hsize, mpi_type, 2, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M4.root->data, hsize*hsize, mpi_type, 3, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M5.root->data, hsize*hsize, mpi_type, 4, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M6.root->data, hsize*hsize, mpi_type, 5, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					// MPI_Recv(M7.root->data, hsize*hsize, mpi_type, 3, 16, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				if (rank ==1)
				{
					MPI_Send(M2.root->data, hsize*hsize, mpi_type, 0, 11, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank ==2)
				{
					MPI_Send(M3.root->data, hsize*hsize, mpi_type, 0, 12, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank ==3)
				{
					MPI_Send(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank == 4)
				{
					MPI_Send(M5.root->data, hsize*hsize, mpi_type, 0, 14, MPI_COMM_WORLD);
				}
				else if (rank ==5)
				{
					MPI_Send(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD);
				}
			}
			else if (procs == 7)
			{
				if (rank ==0)
				{
					MPI_Recv(M2.root->data, hsize*hsize, mpi_type, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M3.root->data, hsize*hsize, mpi_type, 2, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M4.root->data, hsize*hsize, mpi_type, 3, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M5.root->data, hsize*hsize, mpi_type, 4, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M6.root->data, hsize*hsize, mpi_type, 5, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M7.root->data, hsize*hsize, mpi_type, 6, 16, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				if (rank ==1)
				{
					MPI_Send(M2.root->data, hsize*hsize, mpi_type, 0, 11, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank ==2)
				{
					MPI_Send(M3.root->data, hsize*hsize, mpi_type, 0, 12, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank ==3)
				{
					MPI_Send(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank == 4)
				{
					MPI_Send(M5.root->data, hsize*hsize, mpi_type, 0, 14, MPI_COMM_WORLD);
				}
				else if (rank ==5)
				{
					MPI_Send(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD);
				}
				else if (rank ==6)
				{
					MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
			}
		}
		if (procs ==8)
		{
			MPI_Request req;
			switch (procs)
			{
				case 8:
					switch (rank%7)
					{
						case 0:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B11);
							StrassenRecursiveM1_2(M1, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 1:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A10, *A11);
							StrassenRecursiveLocal(M2, Temp1, *B00);
							MPI_Isend(M2.root->data, hsize*hsize, mpi_type, 0, 11, MPI_COMM_WORLD, &req);
							MPI_Wait(&req, MPI_STATUS_IGNORE);
							delete Temp1.root;
							break;
						}
						case 2:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *B01, *B11);
							StrassenRecursiveLocal(M3, *A00, Temp1);
							MPI_Isend(M3.root->data, hsize*hsize, mpi_type, 0, 12, MPI_COMM_WORLD, &req);
							MPI_Wait(&req, MPI_STATUS_IGNORE);
							delete Temp1.root;
							break;
						}
						case 3:
						{
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp2, *B10, *B00);
							StrassenRecursiveLocal(M4, *A11, Temp2);
							MPI_Isend(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD, &req);
							MPI_Wait(&req, MPI_STATUS_IGNORE);
							delete Temp2.root;
							break;
						}
						case 4:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
							StrassenRecursiveLocal(M5, Temp1, *B11);
							MPI_Isend(M5.root->data, hsize*hsize, mpi_type, 0, 14, MPI_COMM_WORLD, &req);
							MPI_Wait(&req, MPI_STATUS_IGNORE);
							delete Temp1.root;
							break;
						}	
						case 5:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A10, *A00);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B01);
							StrassenRecursiveLocal(M6, Temp1, Temp2);
							MPI_Isend(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD, &req);
							MPI_Wait(&req, MPI_STATUS_IGNORE);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 6:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A01, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B10, *B11);
							StrassenRecursiveLocal(M7, Temp1, Temp2);
							MPI_Isend(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD, &req);
							MPI_Wait(&req, MPI_STATUS_IGNORE);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
					}
					break;

			}
			if (rank == 0)
			{
				
				MPI_Request req[6];
				// MPI_Irecv(M1.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &req[0]);
				MPI_Irecv(M2.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 11, MPI_COMM_WORLD, &req[1]);
				MPI_Irecv(M3.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 12, MPI_COMM_WORLD, &req[2]);
				MPI_Irecv(M4.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 13, MPI_COMM_WORLD, &req[3]);
				MPI_Irecv(M5.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 14, MPI_COMM_WORLD, &req[4]);
				MPI_Irecv(M6.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 15, MPI_COMM_WORLD, &req[5]);
				MPI_Irecv(M7.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 16, MPI_COMM_WORLD, &req[0]);
				MPI_Waitall(6, req, MPI_STATUSES_IGNORE);

			}
			
		}

		if (rank == 0)
		{
			SplittableMatrix<T>* C00 = C.split(0, 0);
			SplittableMatrix<T>* C01 = C.split(0, 1);
			SplittableMatrix<T>* C10 = C.split(1, 0);
			SplittableMatrix<T>* C11 = C.split(1, 1);

			ufunc::addition::Seq<T>::operate(*C11, M3, M6);
			// delete M6.root;
			
			ufunc::addition::Seq<T>::operate(*C00, M4, M7);
			// delete M7.root;

			ufunc::addition::Seq<T>::operate(*C10, M2, M4);
			// delete M4.root;
			
			ufunc::addition::Seq<T>::operate(*C01, M3, M5);
			// delete M3.root;
			ufunc::addition::Seq<T>::re_operate(*C00, *C00, M5);
			// delete M5.root;
		

			ufunc::addition::Seq<T>::re_operate(*C11, *C11, M2);

			ufunc::addition::Seq<T>::operate(*C00, *C00, M1);
			ufunc::addition::Seq<T>::operate(*C11, *C11, M1);
			delete C00;
			delete C01;
			delete C10;
			delete C11;
		}
		delete A00;
		delete A01;
		delete A10;	
		delete A11;
		delete B00;
		delete B01;
		delete B10;
		delete B11;	
		delete M1.root;
		delete M2.root;
		delete M3.root;
		delete M4.root;
		delete M5.root;
		delete M6.root;
		delete M7.root;	
	}
};

template<class T>
size_t HybridStrassen<T>::threshold = 32;

template<class T>
class MPIStrassen {
public:
	// MPI_Datatype mpi_type = sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE;

	// static std::atomic<bool> g_stop;
	static inline void operate(SplittableMatrix<T>& out,
                           const SplittableMatrix<T>& lhs,
                           const SplittableMatrix<T>& rhs)
	{
		int rank, nprocs;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

		// 1) Broadcast N (dimension) để mọi rank đồng nhất
		int N;
		if (rank == 0) N = (int)lhs.cdim;  // giả định lhs vuông
		MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// 2) Đảm bảo worker có buffer đúng size trước khi nhận data
		SplittableMatrix<T> A = lhs;
		SplittableMatrix<T> B = rhs;

		if (rank != 0) {
			A = SplittableMatrix<T>(new Buffer<T>(N), N);
			B = SplittableMatrix<T>(new Buffer<T>(N), N);

			// out cũng phải cùng size để lát Bcast kết quả
			out = SplittableMatrix<T>(new Buffer<T>(N), N);
		}

		// 3) Bcast data với count giống nhau trên mọi rank
		MPI_Bcast(A.root->data, N*N, mpi_type, 0, MPI_COMM_WORLD);
		MPI_Bcast(B.root->data, N*N, mpi_type, 0, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);

		// >>> từ đây trở đi dùng A,B thay vì lhs,rhs <<<
		if (rank == 0) {
			if ((size_t)N <= threshold) {
				Seq<T>::operate(out, A, B);
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(out.root->data, N*N, mpi_type, 0, MPI_COMM_WORLD);
				return;
			}
			StrassenRecursive(out, A, B);
		} else {
			if ((size_t)N <= threshold) {
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(out.root->data, N*N, mpi_type, 0, MPI_COMM_WORLD);
				return;
			}
			StrassenRecursive(out, A, B);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(out.root->data, N*N, mpi_type, 0, MPI_COMM_WORLD);
		// if (rank ==0)
		// {
		// 	SplittableMatrix<T> checkC(new Buffer<T>(N),N);
		// 	Seq<T>::operate(checkC,lhs,rhs);
		// 	SplittableMatrix<T>::is_same(checkC, out);
		// }
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
	static inline MPI_Datatype mpi_type = sizeof(T) == 4 ? MPI_FLOAT : MPI_DOUBLE;


	MPIStrassen(SplittableMatrix<T>* out, const SplittableMatrix<T>* lhs, const SplittableMatrix<T>* rhs)
		: out(out), lhs(lhs), rhs(rhs) {}


	static void StrassenRecursiveLocal(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B)
	{
		const size_t N = C.cdim;
		if (N <= threshold) {
			Seq<T>::operate(C, A, B);
			return;
		}

		size_t hsize = N / 2;

		SplittableMatrix<T> M1(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> M2(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> M3(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> M4(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> M5(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> M6(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> M7(new Buffer<T>(hsize), hsize);

		SplittableMatrix<T>* A00 = A.split(0,0);
		SplittableMatrix<T>* A01 = A.split(0,1);
		SplittableMatrix<T>* A10 = A.split(1,0);
		SplittableMatrix<T>* A11 = A.split(1,1);

		SplittableMatrix<T>* B00 = B.split(0,0);
		SplittableMatrix<T>* B01 = B.split(0,1);
		SplittableMatrix<T>* B10 = B.split(1,0);
		SplittableMatrix<T>* B11 = B.split(1,1);

		SplittableMatrix<T>* C00 = C.split(0,0);
		SplittableMatrix<T>* C01 = C.split(0,1);
		SplittableMatrix<T>* C10 = C.split(1,0);
		SplittableMatrix<T>* C11 = C.split(1,1);

		SplittableMatrix<T> T1(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> T2(new Buffer<T>(hsize), hsize);

		ufunc::addition::Seq<T>::operate(T1, *A00, *A11);
		ufunc::addition::Seq<T>::operate(T2, *B00, *B11);
		StrassenRecursiveLocal(M1, T1, T2);

		ufunc::addition::Seq<T>::operate(T1, *A10, *A11);
		StrassenRecursiveLocal(M2, T1, *B00);

		ufunc::addition::Seq<T>::re_operate(T2, *B01, *B11);
		StrassenRecursiveLocal(M3, *A00, T2);

		ufunc::addition::Seq<T>::re_operate(T2, *B10, *B00);
		StrassenRecursiveLocal(M4, *A11, T2);

		ufunc::addition::Seq<T>::operate(T1, *A00, *A01);
		StrassenRecursiveLocal(M5, T1, *B11);

		ufunc::addition::Seq<T>::re_operate(T1, *A10, *A00);
		ufunc::addition::Seq<T>::operate(T2, *B00, *B01);
		StrassenRecursiveLocal(M6, T1, T2);

		ufunc::addition::Seq<T>::re_operate(T1, *A01, *A11);
		ufunc::addition::Seq<T>::operate(T2, *B10, *B11);
		StrassenRecursiveLocal(M7, T1, T2);

		delete T1.root;
		delete T2.root;

		// combine M1..M7 into C00..C11 giống bản đúng của bạn
		ufunc::addition::Seq<T>::operate(*C11, M3, M6);
		delete M6.root;

		ufunc::addition::Seq<T>::operate(*C00, M4, M7);
		delete M7.root;

		ufunc::addition::Seq<T>::operate(*C10, M2, M4);
		delete M4.root;

		ufunc::addition::Seq<T>::operate(*C01, M3, M5);
		delete M3.root;
		
		ufunc::addition::Seq<T>::re_operate(*C00, *C00, M5);
		delete M5.root;

		ufunc::addition::Seq<T>::re_operate(*C11, *C11, M2);
		delete M2.root;

		ufunc::addition::Seq<T>::operate(*C00, *C00, M1);
		ufunc::addition::Seq<T>::operate(*C11, *C11, M1);
		delete M1.root;

		delete A00; delete A01; delete A10; delete A11;
		delete B00; delete B01; delete B10; delete B11;
		delete C00; delete C01; delete C10; delete C11;
	}

	static void StrassenRecursive_group2(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B, MPI_Comm subcomm, int i)
	{
		int sub_rank, sub_size;
		int world_rank;
		MPI_Comm_rank(subcomm, &sub_rank);
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		MPI_Comm_size(subcomm, &sub_size);

		const size_t N = C.cdim;
		if (N <= threshold) {
			Seq<T>::operate(C, A, B);
			return;
		}

		size_t h = N/2;

		// Allocate 7 blocks
		SplittableMatrix<T> M1(new Buffer<T>(h),h);
		SplittableMatrix<T> M2(new Buffer<T>(h),h);
		SplittableMatrix<T> M3(new Buffer<T>(h),h);
		SplittableMatrix<T> M4(new Buffer<T>(h),h);
		SplittableMatrix<T> M5(new Buffer<T>(h),h);
		SplittableMatrix<T> M6(new Buffer<T>(h),h);
		SplittableMatrix<T> M7(new Buffer<T>(h),h);

		SplittableMatrix<T>* A00=A.split(0,0);
		SplittableMatrix<T>* A01=A.split(0,1);
		SplittableMatrix<T>* A10=A.split(1,0);
		SplittableMatrix<T>* A11=A.split(1,1);
		SplittableMatrix<T>* B00=B.split(0,0);
		SplittableMatrix<T>* B01=B.split(0,1);
		SplittableMatrix<T>* B10=B.split(1,0);
		SplittableMatrix<T>* B11=B.split(1,1);

		SplittableMatrix<T> T1(new Buffer<T>(h),h);
		SplittableMatrix<T> T2(new Buffer<T>(h),h);

		if (sub_rank == 0)
		{
			// Compute M1-M4
			ufunc::addition::Seq<T>::operate(T1,*A00,*A11);
			ufunc::addition::Seq<T>::operate(T2,*B00,*B11);
			StrassenRecursiveLocal(M1,T1,T2);

			ufunc::addition::Seq<T>::operate(T1,*A10,*A11);
			StrassenRecursiveLocal(M2,T1,*B00);

			ufunc::addition::Seq<T>::re_operate(T1,*B01,*B11);
			StrassenRecursiveLocal(M3,*A00,T1);

			ufunc::addition::Seq<T>::re_operate(T2,*B10,*B00);
			StrassenRecursiveLocal(M4,*A11,T2);

			// Receive M5-M7 from sub_rank 1
			MPI_Recv(M5.root->data, h*h, mpi_type, 1, 14, subcomm, MPI_STATUS_IGNORE);
			MPI_Recv(M6.root->data, h*h, mpi_type, 1, 15, subcomm, MPI_STATUS_IGNORE);
			MPI_Recv(M7.root->data, h*h, mpi_type, 1, 16, subcomm, MPI_STATUS_IGNORE);

			SplittableMatrix<T>* C00=C.split(0,0);
			SplittableMatrix<T>* C01=C.split(0,1);
			SplittableMatrix<T>* C10=C.split(1,0); 
			SplittableMatrix<T>* C11=C.split(1,1);

			ufunc::addition::Seq<T>::operate(*C11, M3, M6);
			// delete M6.root;
			
			ufunc::addition::Seq<T>::operate(*C00, M4, M7);
			// delete M7.root;

			ufunc::addition::Seq<T>::operate(*C10, M2, M4);
			// delete M4.root;
			
			ufunc::addition::Seq<T>::operate(*C01, M3, M5);
			// delete M3.root;
			ufunc::addition::Seq<T>::re_operate(*C00, *C00, M5);
			// delete M5.root;
		
			ufunc::addition::Seq<T>::re_operate(*C11, *C11, M2);
			// delete M2.root;

			ufunc::addition::Seq<T>::operate(*C00, *C00, M1);
			ufunc::addition::Seq<T>::operate(*C11, *C11, M1);
			delete C00;
			delete C01;
			delete C10;
			delete C11;

		}
		else 
		{
			// Compute M5-M7
			ufunc::addition::Seq<T>::operate(T1,*A00,*A01);
			StrassenRecursiveLocal(M5,T1,*B11);

			ufunc::addition::Seq<T>::re_operate(T1,*A10,*A00);
			ufunc::addition::Seq<T>::operate(T2,*B00,*B01);
			StrassenRecursiveLocal(M6,T1,T2);

			ufunc::addition::Seq<T>::re_operate(T1,*A01,*A11);
			ufunc::addition::Seq<T>::operate(T2,*B10,*B11);
			StrassenRecursiveLocal(M7,T1,T2);
			// Send M5-M7 to sub_rank 0
			MPI_Send(M5.root->data,h*h,mpi_type,0,14,subcomm);
			MPI_Send(M6.root->data,h*h,mpi_type,0,15,subcomm);
			MPI_Send(M7.root->data,h*h,mpi_type,0,16,subcomm);
		}
				MPI_Request req = MPI_REQUEST_NULL;
		if (i == 1)
		{
			MPI_Bcast(C.root->data, N*N, mpi_type, 0, subcomm);
			// if (world_rank == 7)
			// {
			// 	MPI_Send(C.root->data, N*N, mpi_type, 0, 10, MPI_COMM_WORLD);
			// }
		}
		else if (i==2)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 11, MPI_COMM_WORLD, &req);
			}
		}
		else if (i == 3)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 12, MPI_COMM_WORLD, &req);
			}
		}
		else if (i==4)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 13, MPI_COMM_WORLD, &req);
			}
		}
		else if (i==5)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 14, MPI_COMM_WORLD, &req);
			}
		}
		else if (i==6)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 15, MPI_COMM_WORLD, &req);
			}
		}
		else if (i==7)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 16, MPI_COMM_WORLD, &req);
			}
		}
		MPI_Wait(&req, MPI_STATUS_IGNORE);
		delete M1.root;
		delete M2.root;
		delete M3.root;
		delete M4.root;
		delete M5.root;
		delete M6.root;
		delete M7.root;
		delete T1.root;
		delete T2.root;
		delete A00;
		delete A01;
		delete A10;
		delete A11;
		delete B00;
		delete B01;
		delete B10;
		delete B11;

	}

	static void StrassenRecursive_group3(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B, MPI_Comm subcomm, int i )
	{
		int sub_rank, world_rank;
		MPI_Comm_rank(subcomm, &sub_rank);
		// MPI_Comm_size(subcomm, &sub_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

		const size_t N = C.cdim;
		size_t h = N/2;


		SplittableMatrix<T> M1(new Buffer<T>(h),h);
		SplittableMatrix<T> M2(new Buffer<T>(h),h);
		SplittableMatrix<T> M3(new Buffer<T>(h),h);
		SplittableMatrix<T> M4(new Buffer<T>(h),h);
		SplittableMatrix<T> M5(new Buffer<T>(h),h);
		SplittableMatrix<T> M6(new Buffer<T>(h),h);
		SplittableMatrix<T> M7(new Buffer<T>(h),h);

		SplittableMatrix<T>* A00=A.split(0,0);
		SplittableMatrix<T>* A01=A.split(0,1);
		SplittableMatrix<T>* A10=A.split(1,0);
		SplittableMatrix<T>* A11=A.split(1,1);
		SplittableMatrix<T>* B00=B.split(0,0);
		SplittableMatrix<T>* B01=B.split(0,1);
		SplittableMatrix<T>* B10=B.split(1,0);
		SplittableMatrix<T>* B11=B.split(1,1);

		SplittableMatrix<T> T1(new Buffer<T>(h),h);
		SplittableMatrix<T> T2(new Buffer<T>(h),h);

		if (sub_rank == 0)
		{

			ufunc::addition::Seq<T>::operate(T1, *A00, *A11);
			ufunc::addition::Seq<T>::operate(T2, *B00, *B11);
			StrassenRecursiveLocal(M1, T1, T2);

			ufunc::addition::Seq<T>::operate(T1, *A10, *A11);
			StrassenRecursiveLocal(M2, T1, *B00);


			MPI_Recv(M3.root->data, h*h, mpi_type, 1, 12, subcomm, MPI_STATUS_IGNORE);
			MPI_Recv(M4.root->data, h*h, mpi_type, 1, 13, subcomm, MPI_STATUS_IGNORE);
			MPI_Recv(M5.root->data, h*h, mpi_type, 1, 14, subcomm, MPI_STATUS_IGNORE);


			MPI_Recv(M6.root->data, h*h, mpi_type, 2, 15, subcomm, MPI_STATUS_IGNORE);
			MPI_Recv(M7.root->data, h*h, mpi_type, 2, 16, subcomm, MPI_STATUS_IGNORE);


			SplittableMatrix<T>* C00=C.split(0,0);
			SplittableMatrix<T>* C01=C.split(0,1);
			SplittableMatrix<T>* C10=C.split(1,0);
			SplittableMatrix<T>* C11=C.split(1,1);

			// Standard Strassen combine
			ufunc::addition::Seq<T>::operate(*C11, M3, M6);
			

			ufunc::addition::Seq<T>::operate(*C00, M4, M7);
			

			ufunc::addition::Seq<T>::operate(*C10, M2, M4);
			

			ufunc::addition::Seq<T>::operate(*C01, M3, M5);
			
			ufunc::addition::Seq<T>::re_operate(*C00, *C00, M5);
			

			ufunc::addition::Seq<T>::re_operate(*C11, *C11, M2);
			

			ufunc::addition::Seq<T>::operate(*C00, *C00, M1);
			ufunc::addition::Seq<T>::operate(*C11, *C11, M1);
			

			delete C00;
			delete C01;
			delete C10;
			delete C11;
		}

		else if (sub_rank == 1)
		{
			// M3
			ufunc::addition::Seq<T>::re_operate(T1, *B01, *B11);
			StrassenRecursiveLocal(M3, *A00, T1);

			// M4
			ufunc::addition::Seq<T>::re_operate(T2, *B10, *B00);
			StrassenRecursiveLocal(M4, *A11, T2);

			// M5
			ufunc::addition::Seq<T>::operate(T1, *A00, *A01);
			StrassenRecursiveLocal(M5, T1, *B11);

			// Send M3..M5
			MPI_Send(M3.root->data, h*h, mpi_type, 0, 12, subcomm);
			MPI_Send(M4.root->data, h*h, mpi_type, 0, 13, subcomm);
			MPI_Send(M5.root->data, h*h, mpi_type, 0, 14, subcomm);

		}

		else if (sub_rank == 2)
		{
			// M6
			ufunc::addition::Seq<T>::re_operate(T1,*A10,*A00);
			ufunc::addition::Seq<T>::operate(T2,*B00,*B01);
			StrassenRecursiveLocal(M6,T1,T2);

			// M7
			ufunc::addition::Seq<T>::re_operate(T1,*A01,*A11);
			ufunc::addition::Seq<T>::operate(T2,*B10,*B11);
			StrassenRecursiveLocal(M7,T1,T2);

			// Send M6, M7
			MPI_Send(M6.root->data, h*h, mpi_type, 0, 15, subcomm);
			MPI_Send(M7.root->data, h*h, mpi_type, 0, 16, subcomm);
		}
				MPI_Request req = MPI_REQUEST_NULL;
		if (i == 1)
		{
			MPI_Bcast(C.root->data, N*N, mpi_type, 0, subcomm);
			// if (world_rank == 7)
			// {
			// 	MPI_Send(C.root->data, N*N, mpi_type, 0, 10, MPI_COMM_WORLD);
			// }
		}
		else if (i==2)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 11, MPI_COMM_WORLD, &req);
			}
		}
		else if (i == 3)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 12, MPI_COMM_WORLD, &req);
			}
		}
		else if (i==4)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 13, MPI_COMM_WORLD, &req);
			}
		}
		else if (i==5)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 14, MPI_COMM_WORLD, &req);
			}
		}
		else if (i==6)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 15, MPI_COMM_WORLD, &req);
			}
		}
		else if (i==7)
		{
			if (sub_rank==0)
			{
				MPI_Isend(C.root->data, N*N, mpi_type, 0, 16, MPI_COMM_WORLD, &req);
			}
		}
		MPI_Wait(&req, MPI_STATUS_IGNORE);
		delete M1.root;
		delete M2.root;
		delete M3.root;
		delete M4.root;
		delete M5.root;
		delete M6.root;
		delete M7.root;
		delete T1.root;
		delete T2.root;
		delete A00;
		delete A01;
		delete A10;
		delete A11;
		delete B00;
		delete B01;
		delete B10;
		delete B11;
		
	}


	static void StrassenRecursiveM1_2(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B)
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (A.cdim <= threshold)
		{
			if (rank == 0)
			{
				Seq<T>::operate(C, A, B);
			}
			return;
		}

		int color = (rank == 0 || rank == 7) ? 1 : MPI_UNDEFINED;
		MPI_Group world_group, sub_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		int ranks[2] = {0, 7};
		MPI_Group_incl(world_group, 2, ranks, &sub_group);

		MPI_Comm subcomm = MPI_COMM_NULL;

		MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &subcomm);

		if (subcomm != MPI_COMM_NULL) {
			StrassenRecursive_group2(C, A, B, subcomm, 1);
			MPI_Barrier(subcomm);
			MPI_Comm_free(&subcomm);
		}
		MPI_Group_free(&sub_group);
		MPI_Group_free(&world_group);
	}



	static void StrassenRecursiveM2_2(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B)
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (A.cdim <= threshold)
		{
			if (rank == 1)
			{
				Seq<T>::operate(C, A, B);
				MPI_Request req;
				MPI_Isend(C.root->data, A.cdim*A.rdim, mpi_type, 0, 11, MPI_COMM_WORLD, &req);
				MPI_Wait(&req, MPI_STATUS_IGNORE);
			}
			return;
		}

		int color = (rank == 1 || rank == 8) ? 1 : MPI_UNDEFINED;
		MPI_Group world_group, sub_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		int ranks[2] = {1, 8};
		MPI_Group_incl(world_group, 2, ranks, &sub_group);

		MPI_Comm subcomm = MPI_COMM_NULL;

		MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &subcomm);

		if (subcomm != MPI_COMM_NULL) {
			StrassenRecursive_group2(C, A, B, subcomm, 2);
			MPI_Barrier(subcomm);
			MPI_Comm_free(&subcomm);
			// MPI_Comm_free(&subcomm);
		}
		MPI_Group_free(&sub_group);
		MPI_Group_free(&world_group);
	}

	static void StrassenRecursiveM3_2(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B)
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (A.cdim <= threshold)
		{
			if (rank == 2)
			{
				Seq<T>::operate(C, A, B);
				MPI_Request req;
				MPI_Isend(C.root->data, A.cdim*A.rdim, mpi_type, 0, 12, MPI_COMM_WORLD, &req);
				MPI_Wait(&req, MPI_STATUS_IGNORE);
			}
			return;
		}

		int color = (rank == 2 || rank == 9) ? 1 : MPI_UNDEFINED;
		MPI_Group world_group, sub_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		int ranks[2] = {2, 9};
		MPI_Group_incl(world_group, 2, ranks, &sub_group);

		MPI_Comm subcomm = MPI_COMM_NULL;

		MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &subcomm);

		if (subcomm != MPI_COMM_NULL) {
			StrassenRecursive_group2(C, A, B, subcomm, 3);
			// MPI_Comm_free(&subcomm);
			MPI_Barrier(subcomm);
			MPI_Comm_free(&subcomm);
		}
		MPI_Group_free(&sub_group);
		MPI_Group_free(&world_group);
	}

	static void StrassenRecursiveM4_2(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B)
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (A.cdim <= threshold)
		{
			if (rank == 3)
			{
				Seq<T>::operate(C, A, B);
				MPI_Request req;
				MPI_Isend(C.root->data, A.cdim*A.rdim, mpi_type, 0, 13, MPI_COMM_WORLD, &req);
				MPI_Wait(&req, MPI_STATUS_IGNORE);
			}
			return;
		}

		int color = (rank == 3 || rank == 10) ? 1 : MPI_UNDEFINED;
		MPI_Group world_group, sub_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		int ranks[2] = {3, 10};
		MPI_Group_incl(world_group, 2, ranks, &sub_group);

		MPI_Comm subcomm = MPI_COMM_NULL;

		MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &subcomm);

		if (subcomm != MPI_COMM_NULL) {
			StrassenRecursive_group2(C, A, B, subcomm, 4);
			// MPI_Comm_free(&subcomm);
			MPI_Barrier(subcomm);
			MPI_Comm_free(&subcomm);
		}
		MPI_Group_free(&sub_group);
		MPI_Group_free(&world_group);
	}

	static void StrassenRecursiveM5_2(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B)
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (A.cdim <= threshold)
		{
			if (rank == 4)
			{
				Seq<T>::operate(C, A, B);
				MPI_Request req;
				MPI_Isend(C.root->data, A.cdim*A.rdim, mpi_type, 0, 14, MPI_COMM_WORLD, &req);
				MPI_Wait(&req, MPI_STATUS_IGNORE);
			}
			return;
		}

		int color = (rank == 4 || rank == 11) ? 1 : MPI_UNDEFINED;
		MPI_Group world_group, sub_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		int ranks[2] = {4, 11};
		MPI_Group_incl(world_group, 2, ranks, &sub_group);

		MPI_Comm subcomm = MPI_COMM_NULL;

		MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &subcomm);

		if (subcomm != MPI_COMM_NULL) {
			StrassenRecursive_group2(C, A, B, subcomm, 5);
			// MPI_Comm_free(&subcomm);
			MPI_Barrier(subcomm);
			MPI_Comm_free(&subcomm);
		}
		MPI_Group_free(&sub_group);
		MPI_Group_free(&world_group);
	}

	static void StrassenRecursiveM6_2(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B)
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (A.cdim <= threshold)
		{
			if (rank == 5)
			{
				Seq<T>::operate(C, A, B);
				MPI_Request req;
				MPI_Isend(C.root->data, A.cdim*A.rdim, mpi_type, 0, 15, MPI_COMM_WORLD, &req);
				MPI_Wait(&req, MPI_STATUS_IGNORE);
			}
			return;
		}

		int color = (rank == 5 || rank == 12) ? 1 : MPI_UNDEFINED;
		MPI_Group world_group, sub_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		int ranks[2] = {5, 12};
		MPI_Group_incl(world_group, 2, ranks, &sub_group);

		MPI_Comm subcomm = MPI_COMM_NULL;

		MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &subcomm);

		if (subcomm != MPI_COMM_NULL) {
			StrassenRecursive_group2(C, A, B, subcomm, 6);
			// MPI_Comm_free(&subcomm);
			MPI_Barrier(subcomm);
			MPI_Comm_free(&subcomm);
		}
		MPI_Group_free(&sub_group);
		MPI_Group_free(&world_group);
	}
	
	static void StrassenRecursiveM7_2(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B)
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (A.cdim <= threshold)
		{
			if (rank == 6)
			{
				Seq<T>::operate(C, A, B);
				MPI_Request req;
				MPI_Isend(C.root->data, A.cdim*A.rdim, mpi_type, 0, 16, MPI_COMM_WORLD, &req);
				MPI_Wait(&req, MPI_STATUS_IGNORE);
			}
			return;
		}
		int color = (rank == 6 || rank == 13) ? 1 : MPI_UNDEFINED;
		MPI_Group world_group, sub_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		int ranks[2] = {6, 13};
		MPI_Group_incl(world_group, 2, ranks, &sub_group);

		MPI_Comm subcomm = MPI_COMM_NULL;

		MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &subcomm);

		if (subcomm != MPI_COMM_NULL) {
			StrassenRecursive_group2(C, A, B, subcomm, 7);
			// MPI_Comm_free(&subcomm);
			MPI_Barrier(subcomm);
			MPI_Comm_free(&subcomm);
		}
		MPI_Group_free(&sub_group);
		MPI_Group_free(&world_group);
	}

	static void StrassenRecursiveM1_3(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B)
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (A.cdim <= threshold)
		{
			if (rank == 0)
			{
				Seq<T>::operate(C, A, B);
			}
			return;
		}

		int color = (rank == 0 || rank == 7 || rank == 14) ? 1 : MPI_UNDEFINED;
		MPI_Group world_group, sub_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		int ranks[3] = {0, 7, 14};
		MPI_Group_incl(world_group, 3, ranks, &sub_group);

		MPI_Comm subcomm = MPI_COMM_NULL;

		MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &subcomm);

		if (subcomm != MPI_COMM_NULL) {
			StrassenRecursive_group3(C, A, B, subcomm, 1);
			// MPI_Comm_free(&subcomm);
			MPI_Barrier(subcomm);
			MPI_Comm_free(&subcomm);
		}
		MPI_Group_free(&sub_group);
		MPI_Group_free(&world_group);
	}

	static void StrassenRecursiveM2_3(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B)
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if (A.cdim <= threshold)
		{
			if (rank == 1)
			{
				Seq<T>::operate(C, A, B);
				MPI_Request req;
				MPI_Isend(C.root->data, A.cdim*A.rdim, mpi_type, 0, 11, MPI_COMM_WORLD, &req);
				MPI_Wait(&req, MPI_STATUS_IGNORE);
			}
			return;
		}

		int color = (rank == 1 || rank == 8 || rank == 15) ? 1 : MPI_UNDEFINED;
		MPI_Group world_group, sub_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		int ranks[3] = {1, 8, 15};
		MPI_Group_incl(world_group, 3, ranks, &sub_group);

		MPI_Comm subcomm = MPI_COMM_NULL;

		MPI_Comm_create_group(MPI_COMM_WORLD, sub_group, 0, &subcomm);

		if (subcomm != MPI_COMM_NULL) {
			StrassenRecursive_group3(C, A, B, subcomm, 2);
			// MPI_Comm_free(&subcomm);
			MPI_Barrier(subcomm);
			MPI_Comm_free(&subcomm);
		}
		MPI_Group_free(&sub_group);
		MPI_Group_free(&world_group);
	}


		
	static void StrassenRecursive(SplittableMatrix<T>& C, const SplittableMatrix<T>& A, const SplittableMatrix<T>& B) 
	{
        // const size_t N = std::max(C.rdim, C.cdim);
		const size_t N = C.cdim;
        if (N <= threshold) 
		{
			// CODE_FOR_DEBUG_MODE(printf("Rank: size %zu handled locally\n",  N);)
            Seq<T>::operate(C, A, B);
            return;
        }
		size_t hsize = N / 2;
		int rank = -1;
		int procs = -1;
		MPI_Comm_rank(MPI_COMM_WORLD,&rank); 
		MPI_Comm_size(MPI_COMM_WORLD,&procs);

        // split A, B, C into 4 submatrices using SplittableMatrix::split or view()
		SplittableMatrix<T> M1(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M2(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M3(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> M4(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M5(new Buffer<T>(hsize), hsize);
    	SplittableMatrix<T> M6(new Buffer<T>(hsize), hsize);
		SplittableMatrix<T> M7(new Buffer<T>(hsize), hsize);

		SplittableMatrix<T>* A00 = A.split(0, 0);
		SplittableMatrix<T>* A01 = A.split(0, 1);
		SplittableMatrix<T>* A10 = A.split(1, 0);
		SplittableMatrix<T>* A11 = A.split(1, 1);
		SplittableMatrix<T>* B00 = B.split(0, 0);
		SplittableMatrix<T>* B01 = B.split(0, 1);
		SplittableMatrix<T>* B10 = B.split(1, 0);
		SplittableMatrix<T>* B11 = B.split(1, 1);

		if (procs <= 7)
		{
			for (int i =0;i <7;i++)
			{
				int group = i%procs;
				if (rank == group)
				{
					switch (i)
					{
						case 0:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B11);
							StrassenRecursiveLocal(M1, Temp1, Temp2);
							// MPI_Send(M1.root->data, hsize*hsize, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 1:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A10, *A11);
							StrassenRecursiveLocal(M2, Temp1, *B00);
							// if (rank != 0)
							// 	MPI_Send(M2.root->data, hsize*hsize, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD);
							delete Temp1.root;
							break;
						}
						case 2:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *B01, *B11);
							StrassenRecursiveLocal(M3, *A00, Temp1);
								// if (rank != 0)
								// 	MPI_Send(M3.root->data, hsize*hsize, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
							// MPI_Send(M3.root->data, hsize*hsize, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
							delete Temp1.root;
							break;
						}
						case 3:
						{
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp2, *B10, *B00);
							StrassenRecursiveLocal(M4, *A11, Temp2);
							// if (rank != 0)
							// 	MPI_Send(M4.root->data, hsize*hsize, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD);
							delete Temp2.root;
							break;
						}
						case 4:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
							StrassenRecursiveLocal(M5, Temp1, *B11);
							// if (rank != 0)
							// 	MPI_Send(M5.root->data, hsize*hsize, MPI_DOUBLE, 0, 14, MPI_COMM_WORLD);
							delete Temp1.root;
							break;
						}
						case 5:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A10, *A00);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B01);
							StrassenRecursiveLocal(M6, Temp1, Temp2);
							// if (rank != 0)
							// 	MPI_Send(M6.root->data, hsize*hsize, MPI_DOUBLE, 0, 15, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 6:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
							StrassenRecursiveLocal(M5, Temp1, *B11);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A01, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B10, *B11);
							StrassenRecursiveLocal(M7, Temp1, Temp2);
							// if (rank != 0)
							// 	MPI_Send(M7.root->data, hsize*hsize, MPI_DOUBLE, 0, 16, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
					}
				}

			}
			if (procs == 2)
			{
				if (rank == 0)
				{
					MPI_Recv(M2.root->data, hsize*hsize, mpi_type, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M4.root->data, hsize*hsize, mpi_type, 1, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M6.root->data, hsize*hsize, mpi_type, 1, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				else if (rank ==1)
				{
					MPI_Send(M2.root->data, hsize*hsize, mpi_type, 0, 11, MPI_COMM_WORLD);
					MPI_Send(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD);
					MPI_Send(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD);

				}
			}
			else if (procs ==3)
			{
				if (rank == 0)
				{
					MPI_Recv(M2.root->data, hsize*hsize, mpi_type, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M3.root->data, hsize*hsize, mpi_type, 2, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					// MPI_Recv(M4.root->data, hsize*hsize, mpi_type, 1, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M5.root->data, hsize*hsize, mpi_type, 1, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M6.root->data, hsize*hsize, mpi_type, 2, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					// MPI_Recv(M7.root->data, hsize*hsize, mpi_type, 2, 16, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				if (rank ==1)
				{
					MPI_Send(M2.root->data, hsize*hsize, mpi_type, 0, 11, MPI_COMM_WORLD);
					// MPI_Send(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD);
					MPI_Send(M5.root->data, hsize*hsize, mpi_type, 0, 14, MPI_COMM_WORLD);

				}
				else if (rank ==2)
				{
					MPI_Send(M3.root->data, hsize*hsize, mpi_type, 0, 12, MPI_COMM_WORLD);
					MPI_Send(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
			}
			else if (procs == 4)
			{
				if (rank ==0)
				{
					MPI_Recv(M2.root->data, hsize*hsize, mpi_type, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M3.root->data, hsize*hsize, mpi_type, 2, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M4.root->data, hsize*hsize, mpi_type, 3, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					// MPI_Recv(M5.root->data, hsize*hsize, mpi_type, 1, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M6.root->data, hsize*hsize, mpi_type, 1, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M7.root->data, hsize*hsize, mpi_type, 2, 16, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				if (rank ==1)
				{
					MPI_Send(M2.root->data, hsize*hsize, mpi_type, 0, 11, MPI_COMM_WORLD);
					MPI_Send(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD);
				}
				else if (rank ==2)
				{
					MPI_Send(M3.root->data, hsize*hsize, mpi_type, 0, 12, MPI_COMM_WORLD);
					MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank ==3)
				{
					MPI_Send(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
			}
			else if (procs == 5)
			{
				if (rank ==0)
				{
					MPI_Recv(M2.root->data, hsize*hsize, mpi_type, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M3.root->data, hsize*hsize, mpi_type, 2, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M4.root->data, hsize*hsize, mpi_type, 3, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M5.root->data, hsize*hsize, mpi_type, 4, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					// MPI_Recv(M6.root->data, hsize*hsize, mpi_type, 2, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M7.root->data, hsize*hsize, mpi_type, 1, 16, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				if (rank ==1)
				{
					MPI_Send(M2.root->data, hsize*hsize, mpi_type, 0, 11, MPI_COMM_WORLD);
					MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank ==2)
				{
					MPI_Send(M3.root->data, hsize*hsize, mpi_type, 0, 12, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank ==3)
				{
					MPI_Send(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank == 4)
				{
					MPI_Send(M5.root->data, hsize*hsize, mpi_type, 0, 14, MPI_COMM_WORLD);
				}
		
			}
			else if (procs == 6)
			{
				if (rank ==0)
				{
					MPI_Recv(M2.root->data, hsize*hsize, mpi_type, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M3.root->data, hsize*hsize, mpi_type, 2, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M4.root->data, hsize*hsize, mpi_type, 3, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M5.root->data, hsize*hsize, mpi_type, 4, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M6.root->data, hsize*hsize, mpi_type, 5, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					// MPI_Recv(M7.root->data, hsize*hsize, mpi_type, 3, 16, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				if (rank ==1)
				{
					MPI_Send(M2.root->data, hsize*hsize, mpi_type, 0, 11, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank ==2)
				{
					MPI_Send(M3.root->data, hsize*hsize, mpi_type, 0, 12, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank ==3)
				{
					MPI_Send(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank == 4)
				{
					MPI_Send(M5.root->data, hsize*hsize, mpi_type, 0, 14, MPI_COMM_WORLD);
				}
				else if (rank ==5)
				{
					MPI_Send(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD);
				}
			}
			else if (procs == 7)
			{
				if (rank ==0)
				{
					MPI_Recv(M2.root->data, hsize*hsize, mpi_type, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M3.root->data, hsize*hsize, mpi_type, 2, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M4.root->data, hsize*hsize, mpi_type, 3, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M5.root->data, hsize*hsize, mpi_type, 4, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M6.root->data, hsize*hsize, mpi_type, 5, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(M7.root->data, hsize*hsize, mpi_type, 6, 16, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				if (rank ==1)
				{
					MPI_Send(M2.root->data, hsize*hsize, mpi_type, 0, 11, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank ==2)
				{
					MPI_Send(M3.root->data, hsize*hsize, mpi_type, 0, 12, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank ==3)
				{
					MPI_Send(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD);
					// MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
				else if (rank == 4)
				{
					MPI_Send(M5.root->data, hsize*hsize, mpi_type, 0, 14, MPI_COMM_WORLD);
				}
				else if (rank ==5)
				{
					MPI_Send(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD);
				}
				else if (rank ==6)
				{
					MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
				}
			}
		}
		else if (procs <=14 && procs >7)
		{	
			MPI_Request req;
			switch (procs)
			{
				case 8:
					switch (rank%7)
					{
						case 0:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B11);
							StrassenRecursiveM1_2(M1, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 1:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A10, *A11);
							StrassenRecursiveLocal(M2, Temp1, *B00);
							MPI_Isend(M2.root->data, hsize*hsize, mpi_type, 0, 11, MPI_COMM_WORLD, &req);
							MPI_Wait(&req, MPI_STATUS_IGNORE);
							delete Temp1.root;
							break;
						}
						case 2:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *B01, *B11);
							StrassenRecursiveLocal(M3, *A00, Temp1);
							MPI_Isend(M3.root->data, hsize*hsize, mpi_type, 0, 12, MPI_COMM_WORLD, &req);
							MPI_Wait(&req, MPI_STATUS_IGNORE);
							delete Temp1.root;
							break;
						}
						case 3:
						{
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp2, *B10, *B00);
							StrassenRecursiveLocal(M4, *A11, Temp2);
							MPI_Isend(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD, &req);
							MPI_Wait(&req, MPI_STATUS_IGNORE);
							delete Temp2.root;
							break;
						}
						case 4:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
							StrassenRecursiveLocal(M5, Temp1, *B11);
							MPI_Isend(M5.root->data, hsize*hsize, mpi_type, 0, 14, MPI_COMM_WORLD, &req);
							MPI_Wait(&req, MPI_STATUS_IGNORE);
							delete Temp1.root;
							break;
						}	
						case 5:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A10, *A00);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B01);
							StrassenRecursiveLocal(M6, Temp1, Temp2);
							MPI_Isend(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD, &req);
							 MPI_Wait(&req, MPI_STATUS_IGNORE);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 6:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A01, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B10, *B11);
							StrassenRecursiveLocal(M7, Temp1, Temp2);
							MPI_Isend(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD, &req);
							MPI_Wait(&req, MPI_STATUS_IGNORE);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
					}
					break;
				case 9:
					switch (rank%7)
					{
						case 0:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B11);
							StrassenRecursiveM1_2(M1, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 1:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A10, *A11);
							StrassenRecursiveM2_2(M2, Temp1, *B00);
							delete Temp1.root;
							break;
						}
						case 2:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *B01, *B11);
							StrassenRecursiveLocal(M3, *A00, Temp1);
							MPI_Send(M3.root->data, hsize*hsize, mpi_type, 0, 12, MPI_COMM_WORLD);
							delete Temp1.root;
							break;
						}
						case 3:
						{
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp2, *B10, *B00);
							StrassenRecursiveLocal(M4, *A11, Temp2);
							MPI_Send(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD);
							delete Temp2.root;
							break;
						}
						case 4:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
							StrassenRecursiveLocal(M5, Temp1, *B11);
							MPI_Send(M5.root->data, hsize*hsize, mpi_type, 0, 14, MPI_COMM_WORLD);
							delete Temp1.root;
							break;
						}
						case 5:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A10, *A00);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B01);
							StrassenRecursiveLocal(M6, Temp1, Temp2);
							MPI_Send(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 6:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A01, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B10, *B11);
							StrassenRecursiveLocal(M7, Temp1, Temp2);
							MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
					}
					break;
				case 10:
					switch (rank%7)
					{
						case 0:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B11);
							StrassenRecursiveM1_2(M1, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 1:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A10, *A11);
							StrassenRecursiveM2_2(M2, Temp1, *B00);
							delete Temp1.root;
							break;
						}
						case 2:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *B01, *B11);
							StrassenRecursiveM3_2(M3, *A00, Temp1);
							delete Temp1.root;
							break;
						}
						case 3:
						{
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp2, *B10, *B00);
							StrassenRecursiveLocal(M4, *A11, Temp2);
							MPI_Send(M4.root->data, hsize*hsize, mpi_type, 0, 13, MPI_COMM_WORLD);
							delete Temp2.root;
							break;
						}
						case 4:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
							StrassenRecursiveLocal(M5, Temp1, *B11);
							MPI_Send(M5.root->data, hsize*hsize, mpi_type, 0, 14, MPI_COMM_WORLD);
							delete Temp1.root;
							break;
						}
						case 5:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A10, *A00);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B01);
							StrassenRecursiveLocal(M6, Temp1, Temp2);
							MPI_Send(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 6:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A01, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B10, *B11);
							StrassenRecursiveLocal(M7, Temp1, Temp2);
							MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
					}
					break;
				case 11:
					switch (rank%7)
					{
						case 0:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B11);
							StrassenRecursiveM1_2(M1, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 1:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A10, *A11);
							StrassenRecursiveM2_2(M2, Temp1, *B00);
							delete Temp1.root;
							break;
						}
						case 2:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *B01, *B11);
							StrassenRecursiveM3_2(M3, *A00, Temp1);
							delete Temp1.root;
							break;
						}
						case 3:
						{
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp2, *B10, *B00);
							StrassenRecursiveM4_2(M4, *A11, Temp2);
							delete Temp2.root;
							break;
						}
						case 4:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
							StrassenRecursiveLocal(M5, Temp1, *B11);
							MPI_Send(M5.root->data, hsize*hsize, mpi_type, 0, 14, MPI_COMM_WORLD);
							delete Temp1.root;
							break;
						}
						case 5:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A10, *A00);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B01);
							StrassenRecursiveLocal(M6, Temp1, Temp2);
							MPI_Send(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 6:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A01, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B10, *B11);
							StrassenRecursiveLocal(M7, Temp1, Temp2);
							MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
					}
					break;
				case 12:
					switch (rank%7)
					{
						case 0:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B11);
							StrassenRecursiveM1_2(M1, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 1:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A10, *A11);
							StrassenRecursiveM2_2(M2, Temp1, *B00);
							delete Temp1.root;
							break;
						}
						case 2:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *B01, *B11);
							StrassenRecursiveM3_2(M3, *A00, Temp1);
							delete Temp1.root;
							break;
						}
						case 3:
						{
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp2, *B10, *B00);
							StrassenRecursiveM4_2(M4, *A11, Temp2);
							delete Temp2.root;
							break;
						}
						case 4:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
							StrassenRecursiveM5_2(M5, Temp1, *B11);
							delete Temp1.root;
							break;
						}
						case 5:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A10, *A00);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B01);
							StrassenRecursiveLocal(M6, Temp1, Temp2);
							MPI_Send(M6.root->data, hsize*hsize, mpi_type, 0, 15, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 6:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A01, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B10, *B11);
							StrassenRecursiveLocal(M7, Temp1, Temp2);
							MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
					}
					break;
				case 13:
					switch (rank%7)
					{
						case 0:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B11);
							StrassenRecursiveM1_2(M1, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 1:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A10, *A11);
							StrassenRecursiveM2_2(M2, Temp1, *B00);
							delete Temp1.root;
							break;
						}
						case 2:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *B01, *B11);
							StrassenRecursiveM3_2(M3, *A00, Temp1);
							delete Temp1.root;
							break;
						}
						case 3:
						{
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp2, *B10, *B00);
							StrassenRecursiveM4_2(M4, *A11, Temp2);
							delete Temp2.root;
							break;
						}
						case 4:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
							StrassenRecursiveM5_2(M5, Temp1, *B11);
							delete Temp1.root;
							break;
						}
						case 5:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A10, *A00);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B01);
							StrassenRecursiveM6_2(M6, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 6:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A01, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B10, *B11);
							StrassenRecursiveLocal(M7, Temp1, Temp2);
							MPI_Send(M7.root->data, hsize*hsize, mpi_type, 0, 16, MPI_COMM_WORLD);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}

					}
					break;
				case 14:
					switch (rank%7)
					{
						case 0:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B11);
							StrassenRecursiveM1_2(M1, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 1:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A10, *A11);
							StrassenRecursiveM2_2(M2, Temp1, *B00);
							delete Temp1.root;
							break;
						}
						case 2:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *B01, *B11);
							StrassenRecursiveM3_2(M3, *A00, Temp1);
							delete Temp1.root;
							break;
						}
						case 3:
						{
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp2, *B10, *B00);
							StrassenRecursiveM4_2(M4, *A11, Temp2);
							delete Temp2.root;
							break;
						}
						case 4:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
							StrassenRecursiveM5_2(M5, Temp1, *B11);
							delete Temp1.root;
							break;
						}
						case 5:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A10, *A00);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B01);
							StrassenRecursiveM6_2(M6, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 6:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A01, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B10, *B11);
							StrassenRecursiveM7_2(M7, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
					}
					break;
			}
			if (rank == 0)
			{
				MPI_Request req[6];
				// MPI_Irecv(M1.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &req[0]);
				MPI_Irecv(M2.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 11, MPI_COMM_WORLD, &req[1]);
				MPI_Irecv(M3.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 12, MPI_COMM_WORLD, &req[2]);
				MPI_Irecv(M4.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 13, MPI_COMM_WORLD, &req[3]);
				MPI_Irecv(M5.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 14, MPI_COMM_WORLD, &req[4]);
				MPI_Irecv(M6.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 15, MPI_COMM_WORLD, &req[5]);
				MPI_Irecv(M7.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 16, MPI_COMM_WORLD, &req[0]);
				MPI_Waitall(6, req, MPI_STATUSES_IGNORE);

			}
		}
		else if (procs >=15)
		{
			switch (procs)
			{
				case 15:
					switch (rank%7)
					{
						case 0:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B11);
							StrassenRecursiveM1_3(M1, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 1:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A10, *A11);
							StrassenRecursiveM2_2(M2, Temp1, *B00);
							delete Temp1.root;
							break;
						}
						case 2:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *B01, *B11);
							StrassenRecursiveM3_2(M3, *A00, Temp1);
							delete Temp1.root;
							break;
						}
						case 3:
						{
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp2, *B10, *B00);
							StrassenRecursiveM4_2(M4, *A11, Temp2);
							delete Temp2.root;
							break;
						}
						case 4:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
							StrassenRecursiveM5_2(M5, Temp1, *B11);
							delete Temp1.root;
							break;
						}
						case 5:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A10, *A00);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B01);
							StrassenRecursiveM6_2(M6, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 6:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A01, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B10, *B11);
							StrassenRecursiveM7_2(M7, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
					}
					break;
				case 16:
					switch (rank%7)
					{
						case 0:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B11);
							StrassenRecursiveM1_3(M1, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 1:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A10, *A11);
							StrassenRecursiveM2_3(M2, Temp1, *B00);
							delete Temp1.root;
							break;
						}
						case 2:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *B01, *B11);
							StrassenRecursiveM3_2(M3, *A00, Temp1);
							delete Temp1.root;
							break;
						}
						case 3:
						{
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp2, *B10, *B00);
							StrassenRecursiveM4_2(M4, *A11, Temp2);
							delete Temp2.root;
							break;
						}
						case 4:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::operate(Temp1, *A00, *A01);
							StrassenRecursiveM5_2(M5, Temp1, *B11);
							delete Temp1.root;
							break;
						}
						case 5:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A10, *A00);
							ufunc::addition::Seq<T>::operate(Temp2, *B00, *B01);
							StrassenRecursiveM6_2(M6, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
						case 6:
						{
							SplittableMatrix<T> Temp1(new Buffer<T>(hsize), hsize);
							SplittableMatrix<T> Temp2(new Buffer<T>(hsize), hsize);
							ufunc::addition::Seq<T>::re_operate(Temp1, *A01, *A11);
							ufunc::addition::Seq<T>::operate(Temp2, *B10, *B11);
							StrassenRecursiveM7_2(M7, Temp1, Temp2);
							delete Temp1.root;
							delete Temp2.root;
							break;
						}
					}
					break;
			}
			if (rank == 0)
			{
				MPI_Request req[6];
				// MPI_Irecv(M1.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &req[0]);
				MPI_Irecv(M2.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 11, MPI_COMM_WORLD, &req[1]);
				MPI_Irecv(M3.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 12, MPI_COMM_WORLD, &req[2]);
				MPI_Irecv(M4.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 13, MPI_COMM_WORLD, &req[3]);
				MPI_Irecv(M5.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 14, MPI_COMM_WORLD, &req[4]);
				MPI_Irecv(M6.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 15, MPI_COMM_WORLD, &req[5]);
				MPI_Irecv(M7.root->data, hsize*hsize, mpi_type, MPI_ANY_SOURCE, 16, MPI_COMM_WORLD, &req[0]);
				MPI_Waitall(6, req, MPI_STATUSES_IGNORE);

			}
			
		}

		if (rank == 0)
		{
			SplittableMatrix<T>* C00 = C.split(0, 0);
			SplittableMatrix<T>* C01 = C.split(0, 1);
			SplittableMatrix<T>* C10 = C.split(1, 0);
			SplittableMatrix<T>* C11 = C.split(1, 1);

			ufunc::addition::Seq<T>::operate(*C11, M3, M6);
			// delete M6.root;
			
			ufunc::addition::Seq<T>::operate(*C00, M4, M7);
			// delete M7.root;

			ufunc::addition::Seq<T>::operate(*C10, M2, M4);
			// delete M4.root;
			
			ufunc::addition::Seq<T>::operate(*C01, M3, M5);
			// delete M3.root;
			ufunc::addition::Seq<T>::re_operate(*C00, *C00, M5);
			// delete M5.root;
		

			ufunc::addition::Seq<T>::re_operate(*C11, *C11, M2);

			ufunc::addition::Seq<T>::operate(*C00, *C00, M1);
			ufunc::addition::Seq<T>::operate(*C11, *C11, M1);
			delete C00;
			delete C01;
			delete C10;
			delete C11;
		}
		delete A00;
		delete A01;
		delete A10;	
		delete A11;
		delete B00;
		delete B01;
		delete B10;
		delete B11;	
		delete M1.root;
		delete M2.root;
		delete M3.root;
		delete M4.root;
		delete M5.root;
		delete M6.root;
		delete M7.root;	
	}
};

template<class T>
size_t MPIStrassen<T>::threshold = 32;

#endif // MPI_ENABLE


}; // namespace matmul
}; // namespace ufunc

#endif // MATRIX_UFUNC_MATMUL_H
