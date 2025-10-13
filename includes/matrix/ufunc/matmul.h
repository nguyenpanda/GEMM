#ifndef MATRIX_UFUNC_MATMUL_H
#define MATRIX_UFUNC_MATMUL_H

#include "../viewer.h"
#include "add.h"

namespace ufunc {
namespace matmul {

template<class T>
class Seq {
public:
	static inline void operate(Buffer<T>& out, Buffer<T>& lhs, Buffer<T>& rhs);
	static inline void operate(SplittableMatrix<T>& out, SplittableMatrix<T>& lhs, SplittableMatrix<T>& rhs);

private:
	static inline void compute(Buffer<T>& out, Buffer<T>& lhs, Buffer<T>& rhs);
};

template<class T>
class OmpVanilla {
public:
	static inline void operate(Buffer<T>& out, Buffer<T>& lhs, Buffer<T>& rhs);
	static inline void operate(SplittableMatrix<T>& out, SplittableMatrix<T>& lhs, SplittableMatrix<T>& rhs);

private:
	static inline void compute(Buffer<T>& out, Buffer<T>& lhs, Buffer<T>& rhs);
};

template<class T>
class OmpForkJoin {
public:
	static inline void operate(SplittableMatrix<T>& out, SplittableMatrix<T>& lhs, SplittableMatrix<T>& rhs);

private:
	static size_t threshold;

private:
	static inline void compute(SplittableMatrix<T>& out, SplittableMatrix<T>& lhs, SplittableMatrix<T>& rhs);
};

////////////////////////////////////////////////////////
///////////          Implementation          ///////////
////////////////////////////////////////////////////////

template<class T>
void Seq<T>::compute(Buffer<T>& out, Buffer<T>& lhs, Buffer<T>& rhs) {
	for (size_t i = 0; i < out.size(); i++) {
		out[i] = lhs[i] + rhs[i];
	}
}

template<class T>
void OmpVanilla<T>::compute(Buffer<T>& out, Buffer<T>& lhs, Buffer<T>& rhs) {
	#pragma omp parallel
	{
		#pragma omp for
		for (size_t i = 0; i < out.size(); i++) {
			out[i] = lhs[i] + rhs[i];
		}
	}
}

template<class T>
void ForkJoinOMP<T>::compute(SplittableMatrix<T>& out, SplittableMatrix<T>& lhs, SplittableMatrix<T>& rhs) {
	const size_t N = out.cdim;

	if (N <= threshold) {
		Seq<T>::operate(out, lhs, rhs);
		return;
	}


}

////////////////////////////////////////////////////////
/////////////////          API          ////////////////
////////////////////////////////////////////////////////

template<class T>
void Seq<T>::operate(Buffer<T>& out, Buffer<T>& lhs, Buffer<T>& rhs) {
	compute(out, lhs, rhs);
}

template<class T>
void Seq<T>::operate(SplittableMatrix<T>& out, SplittableMatrix<T>& lhs, SplittableMatrix<T>& rhs) {
	compute(*out, *lhs, *rhs);
}

template<class T>
void OmpVanilla<T>::operate(Buffer<T>& out, Buffer<T>& lhs, Buffer<T>& rhs) {
	compute(out, lhs, rhs);
}

template<class T>
void OmpVanilla<T>::operate(SplittableMatrix<T>& out, SplittableMatrix<T>& lhs, SplittableMatrix<T>& rhs) {
	compute(*out, *lhs, *rhs);
}

template<class T>
void ForkJoinOmp<T>::operate(SplittableMatrix<T>& out, SplittableMatrix<T>& lhs, SplittableMatrix<T>& rhs) {
	#pragma omp parallel
	{
		#pragma omp single
		{
			compute(out, lhs, rhs);	
		}
	}
}

}; // namespace matmul
}; // namespace ufunc

#endif // MATRIX_UFUNC_MATMUL_H
