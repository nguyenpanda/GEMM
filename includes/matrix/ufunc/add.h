#ifndef MATRIX_UFUNC_ADD_H
#define MATRIX_UFUNC_ADD_H

#include "../viewer.h"

namespace ufunc {
namespace addition {

template<class T>
class Seq {
public:
	static inline void operate(Buffer<T>& out, const Buffer<T>& lhs, const Buffer<T>& rhs) {
		#pragma omp simd
		for (size_t i = 0; i < out.size(); i++) {
			out.data[i] = lhs.data[i] + rhs.data[i];
		}
	}

	static inline void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs)  {
		for (size_t i = 0; i < out.rdim; i++) {
			#pragma omp simd
			for (size_t j = 0; j < out.cdim; j++) {
				size_t out_idx = out.map2Dto1DIndex(i, j);
				size_t lhs_idx = lhs.map2Dto1DIndex(i, j);
				size_t rhs_idx = rhs.map2Dto1DIndex(i, j);
				out.root->data[out_idx] = lhs.root->data[lhs_idx] + rhs.root->data[rhs_idx];
			}
		}
	}

		static inline void re_operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs)  {
		for (size_t i = 0; i < out.rdim; i++) {
			#pragma omp simd
			for (size_t j = 0; j < out.cdim; j++) {
				size_t out_idx = out.map2Dto1DIndex(i, j);
				size_t lhs_idx = lhs.map2Dto1DIndex(i, j);
				size_t rhs_idx = rhs.map2Dto1DIndex(i, j);
				out.root->data[out_idx] = lhs.root->data[lhs_idx] - rhs.root->data[rhs_idx];
			}
		}
	}
};


template<class T>
class OmpVanilla {
public:
	static inline void operate(Buffer<T>& out, const Buffer<T>& lhs, const Buffer<T>& rhs) {
		#pragma omp parallel
		{
			#pragma omp for simd
			for (size_t i = 0; i < out.size(); i++) {
				out.data[i] = lhs.data[i] + rhs.data[i];
			}
		}
	}

	static inline void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) {
		#pragma omp parallel
		{
			#pragma omp for simd
			for (size_t i = 0; i < out.rdim; i++) {
				for (size_t j = 0; j < out.cdim; j++) {
					size_t out_idx = out.map2Dto1DIndex(i, j);
					size_t lhs_idx = lhs.map2Dto1DIndex(i, j);
					size_t rhs_idx = rhs.map2Dto1DIndex(i, j);
					out.root->data[out_idx] = lhs.root->data[lhs_idx] + rhs.root->data[rhs_idx];
				}
			}
		}
	}

	static inline void re_operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) {
		#pragma omp parallel
		{
			#pragma omp for simd
			for (size_t i = 0; i < out.rdim; i++) {
				for (size_t j = 0; j < out.cdim; j++) {
					size_t out_idx = out.map2Dto1DIndex(i, j);
					size_t lhs_idx = lhs.map2Dto1DIndex(i, j);
					size_t rhs_idx = rhs.map2Dto1DIndex(i, j);
					out.root->data[out_idx] = lhs.root->data[lhs_idx] - rhs.root->data[rhs_idx];
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

	static size_t get_threshold() {
		return threshold;
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
			"[ufunc][addition][OmpForkJoin] tid = \033[1;95m%d\033[0m " "format = \033[1;95m%-5s\033[0m, " "dim = \033[1;95m%6zu\033[0m, " "rDis = \033[1;95m%6zu\033[0m, " "cDis = \033[1;95m%6zu\033[0m\n", 
			omp_get_thread_num(), format.c_str(), out.rdim, out.rDis / out.rdim, out.cDis / out.cdim);
		)

		const size_t N = out.rdim;
		if (N <= threshold) {
			Seq<T>::operate(out, lhs, rhs);
			return;
		}

		OmpForkJoin<T>* tasks[4];

		#pragma omp unroll full
		for (int i = 0; i < 4; i++) {
			// The following loop must iterate backward.
			// This is because Nguyenpanda designed the ForkJoin model to execute tasks
			// from left to right and from top to bottom.
			tasks[3 - i] = new OmpForkJoin<T>(
				out.split(i >> 1, i & 1),
				lhs.split(i >> 1, i & 1),
				rhs.split(i >> 1, i & 1)
			);
		}

		#pragma omp unroll full
		for (int i = 0; i < 4; ++i) {
			CODE_FOR_DEBUG_MODE(std::string temp = format + std::to_string(3-i);)
			#pragma omp task
			compute(*tasks[i]->out, *tasks[i]->lhs, *tasks[i]->rhs 
				CODE_FOR_DEBUG_MODE(, temp)
			);
		}

		#pragma omp taskwait

		#pragma omp unroll full
		for (int i = 0; i < 4; i++) {
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
class GODDAMNNN {
public:
	static inline void operate(SplittableMatrix<T>& out, const SplittableMatrix<T>& lhs, const SplittableMatrix<T>& rhs) {
		size_t N = static_cast<size_t>(out.cdim);
		if (N <= 128) {
			Seq<T>::operate(out, lhs, rhs);
			return;
		}

		size_t THRESHOLD = OmpForkJoin<T>::get_threshold();
    	OmpForkJoin<T>::set_threshold(static_cast<size_t>(N / 2));
		OmpForkJoin<T>::operate(out, lhs, rhs);
		OmpForkJoin<T>::set_threshold(THRESHOLD);
	}
};


}; // namespace addition
}; // namespace ufunc

#endif // MATRIX_UFUNC_ADD_H
