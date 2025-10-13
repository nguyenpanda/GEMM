#ifndef MATRIX_INITIALIZER_RANDOM_H
#define MATRIX_INITIALIZER_RANDOM_H

#include "../_core/buffer.h"
#include "../viewer.h"

#include <chrono>
#include <cstddef>
#include <random>

#ifdef _OPENMP
	#include <omp.h>
#endif // _OPENMP

template<class T>
class InitializeRandom {
public:
    size_t min, max;
    unsigned seed;
    std::mt19937 generator;

public:
    InitializeRandom(size_t _min, size_t _max);
    InitializeRandom(size_t _min, size_t _max, unsigned _seed);
    ~InitializeRandom() = default;
    void fill(T* data, size_t length) const;
    void fill(Buffer<T>& buffer) const;
    void fill(SplittableMatrix<T>& matrix) const;
};

////////////////////////////////////////////////////////
///////////          Implementation          ///////////
////////////////////////////////////////////////////////

template<class T>
InitializeRandom<T>::InitializeRandom(size_t _min, size_t _max) {
    seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    min = _min;
    max = _max;
    generator = std::mt19937(seed);
}

template<class T>
InitializeRandom<T>::InitializeRandom(size_t _min, size_t _max, unsigned _seed) {
    seed = _seed;
    min = _min;
    max = _max;
    generator = std::mt19937(seed);
}

template<class T>
void InitializeRandom<T>::fill(T* data, size_t length) const {
    #pragma omp parallel firstprivate(length) num_threads(omp_get_num_procs())
    {
        std::uniform_real_distribution<float> distribution(min, max);
        std::mt19937 local_generator = generator;

        #pragma omp for
        for (size_t i = 0; i < length; i++) {
            data[i] = distribution(local_generator);
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
