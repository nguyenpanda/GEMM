#ifndef MATRIX_INITIALIZER_RANDOM_H
#define MATRIX_INITIALIZER_RANDOM_H

#include "../_core/buffer.h"
#include "../viewer.h"

#include <chrono>
#include <cstddef>
#include <random>

template<class T>
class InitializeRandom {
public:
    T min, max;

public:
    InitializeRandom(T _min, T _max);
    ~InitializeRandom() = default;
    void fill(T* data, size_t length) const;
    void fill(Buffer<T>& buffer) const;
    void fill(SplittableMatrix<T>& matrix) const;
};

////////////////////////////////////////////////////////
///////////          Implementation          ///////////
////////////////////////////////////////////////////////

template<class T>
InitializeRandom<T>::InitializeRandom(T _min, T _max) {
    min = _min;
    max = _max;
}

template<class T>
void InitializeRandom<T>::fill(T* data, size_t length) const {
    #pragma omp parallel
    {
        thread_local std::mt19937 generator(
            static_cast<unsigned long>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count() +
                omp_get_thread_num()
            )
        );

        std::uniform_real_distribution<double> distribution(min, max);

        #pragma omp for
        for (size_t i = 0; i < length; ++i) {
            data[i] = static_cast<T>(distribution(generator));
        }
    }
}

template<class T>
void InitializeRandom<T>::fill(Buffer<T>& buffer) const {
    fill(buffer.data, buffer.size());
}

template<class T>
void InitializeRandom<T>::fill(SplittableMatrix<T>& matrix) const {
    if (matrix.root) {
        fill(*matrix.root);
    }
}

#endif // MATRIX_INITIALIZER_RANDOM_H
