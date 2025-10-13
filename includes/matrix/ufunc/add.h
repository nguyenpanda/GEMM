#ifndef MATRIX_UFUNC_ADD_H
#define MATRIX_UFUNC_ADD_H

#include "../viewer.h"
#include <vector>

namespace ufunc {
namespace addition {

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
	void compute();

	static void set_threshold(size_t _threshold);

private:
	static size_t threshold;
	SplittableMatrix<T>* out;
	SplittableMatrix<T>* lhs;
	SplittableMatrix<T>* rhs;

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
void OmpForkJoin<T>::compute(SplittableMatrix<T>& out, SplittableMatrix<T>& lhs, SplittableMatrix<T>& rhs) {
	const size_t N = out.rdim;

	if (N <= threshold) {
		Seq<T>::operate(out, lhs, rhs);
		return;
	}

	OmpForkJoin<T>* tasks[4];

	for (size_t i = 0; i < 4; i++) {
		tasks[i] = new OmpForkJoin<T>(
			out.split(i >> 1, i & 1),
			lhs.split(i >> 1, i & 1),
			rhs.split(i >> 1, i & 1)
		);
	}

	for (size_t i = 0; i < 4; i++) {
		#pragma omp task
		{
			compute(tasks[i]->out, tasks[i]->lhs, tasks[i]->rhs);
		}
		
	}

	#pragma omp taskwait

	for (size_t i = 0; i < 4; i++) {
		delete tasks[i]->lhs;
		delete tasks[i]->rhs;
		delete tasks[i]->out;
		delete tasks[i];
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
void OmpForkJoin<T>::set_threshold(size_t _threshold) {
	threshold = _threshold;
}

template<class T>
void OmpForkJoin<T>::compute() {
	#pragma omp parallel
	{
		#pragma omp single
		{
			compute(*out, *lhs, *rhs);
		}
	}
}

}; // namespace addition
}; // namespace ufunc

#endif // MATRIX_UFUNC_ADD_H
