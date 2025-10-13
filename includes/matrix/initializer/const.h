#ifndef MATRIX_INITIALIZER_CONST_H
#define MATRIX_INITIALIZER_CONST_H

#include "../_core/buffer.h"
#include "../viewer.h"

#include <cstddef>

template<class T>
class InitializeConstant {
public:
    T value;

public:
    explicit InitializeConstant(T value);
    ~InitializeConstant() = default;
    void fill(T* data, size_t length) const;
    void fill(Buffer<T>& buffer) const;
    void fill(SplittableMatrix<T>& matrix) const;
};

////////////////////////////////////////////////////////
///////////          Implementation          ///////////
////////////////////////////////////////////////////////

template<class T>
InitializeConstant<T>::InitializeConstant(T _value) {
    value = _value;
}

template<class T>
void InitializeConstant<T>::fill(T* data, size_t length) const {
    #pragma omp parallel
	{
		#pragma omp for
		for (size_t i = 0; i < length; i++) {
			data[i] = value;
		}
	}
}

template<class T>
void InitializeConstant<T>::fill(Buffer<T>& buffer) const {
    fill(buffer.data, buffer.size());
}

template<class T>
void InitializeConstant<T>::fill(SplittableMatrix<T>& matrix) const {
    if (matrix.root) {
        fill(*matrix.root);
    }
}

#endif // MATRIX_INITIALIZER_CONST_H
