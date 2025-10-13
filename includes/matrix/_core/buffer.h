#ifndef _MATRIX__CORE_BUFFER_H
#define _MATRIX__CORE_BUFFER_H

#include "macro.h"

#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>

template<class T>
class Buffer {
public:
	size_t rdim, cdim;
	T* data;

public:
	Buffer(size_t ndim);
	Buffer(size_t _rdim, size_t _cdim);
	~Buffer();
	inline size_t size() const;
	void print() const;
	static bool is_same(Buffer<T>& lhs, Buffer<T>& rhs);
};

template<class T>
Buffer<T>::Buffer(size_t ndim) {
	rdim = ndim;
	cdim = ndim;
	const size_t size = rdim * cdim;
	size_t nblock = static_cast<size_t>(size / 64);
	const size_t remainder = static_cast<size_t>(size % 64);
	if (remainder >= 1) {
		nblock += 1;
	}
	data = static_cast<T*>(aligned_alloc(64, 64 * nblock * sizeof(T)));
}

template<class T>
Buffer<T>::Buffer(size_t _rdim, size_t _cdim) {
	rdim = _rdim;
	cdim = _cdim;
	const size_t size = rdim * cdim;
	size_t nblock = static_cast<size_t>(size / 64);
	const size_t remainder = static_cast<size_t>(size % 64);
	if (remainder >= 1) {
		nblock += 1;
	}
	data = static_cast<T*>(aligned_alloc(64, 64 * nblock * sizeof(T)));
}

template<class T>
Buffer<T>::~Buffer() {
	delete[] data;
}

template<class T>
inline size_t Buffer<T>::size() const {
	return rdim * cdim;
}

template<class T>
void Buffer<T>::print() const {
	float* buffer = static_cast<float*>(data);
	for (size_t i = 0; i < rdim; i++) {
		for (size_t j = 0; j < cdim; j++) {
			printf("%5.2f ", buffer[i * cdim + j]);
		}
		printf("\n");
	}
}
template<class T>
inline bool Buffer<T>::is_same(Buffer<T>& lhs, Buffer<T>& rhs) {
	if (lhs.rdim != rhs.rdim || lhs.cdim != rhs.cdim) return false;

	for (size_t i = 0; i < lhs.size(); i++) {
		if (lhs.data[i] != rhs.data[i]) {
			return false;
		}
	}
	
	return true;
}

#endif // _MATRIX__CORE_BUFFER_H
