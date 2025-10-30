#ifndef MATRIX_UFUNC_MATMUL_H
#define MATRIX_UFUNC_MATMUL_H

#include "../viewer.h"
#include "add.h"
#include <assert.h>

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
		#elif PRECISION_MODE == PRECISION_KUNAN
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
		#elif PRECISION_MODE == PRECISION_KUNAN
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


}; // namespace matmul
}; // namespace ufunc

#endif // MATRIX_UFUNC_MATMUL_H
