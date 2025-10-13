#ifndef MATRIX_VIEW_SPLITTABLE_H
#define MATRIX_VIEW_SPLITTABLE_H

#include "../_core/buffer.h"

template<class T>
class SplittableMatrix {
public:
	size_t rDis, cDis;
	size_t rdim, cdim;
	Buffer<T>* root;

public:
	SplittableMatrix();
	SplittableMatrix(Buffer<T>* _root, size_t ndim);
	SplittableMatrix(Buffer<T>* _root, size_t _rdim, size_t _cdim);
	SplittableMatrix<T>* split(size_t i, size_t j);
	~SplittableMatrix() = default;
	inline size_t map2Dto1DIndex(size_t i, size_t j) const;
	inline T get(size_t index) const; // USE TRUE INDEX
	inline void set(size_t index, T value); // USE TRUE INDEX
	inline void print() const;
	static bool is_same(SplittableMatrix<T>& lhs, SplittableMatrix<T>& rhs);

private:
	SplittableMatrix(Buffer<T>* _root, 
		size_t _rDis, size_t _cDis, 
		size_t _rdim, size_t _cdim);
};

////////////////////////////////////////////////////////
///////////          Implementation          ///////////
////////////////////////////////////////////////////////

template<class T>
SplittableMatrix<T>::SplittableMatrix()
	: rDis(0), cDis(0), rdim(0), cdim(0), root(nullptr) {}

template<class T>
SplittableMatrix<T>::SplittableMatrix(Buffer<T>* _root, size_t ndim) 
	: rDis(0), cDis(0), rdim(ndim), cdim(ndim), root(_root) {}

template<class T>
SplittableMatrix<T>::SplittableMatrix(Buffer<T>* _root, size_t _rdim, size_t _cdim)
	: rDis(0), cDis(0), rdim(_rdim), cdim(_cdim), root(_root) {}

template<class T>
SplittableMatrix<T>::SplittableMatrix(Buffer<T>* _root, 
	size_t _rDis, size_t _cDis, 
	size_t _rdim, size_t _cdim) 
	: rDis(_rDis), cDis(_cDis), rdim(_rdim), cdim(_cdim), root(_root) {}

template<class T>
SplittableMatrix<T>* SplittableMatrix<T>::split(size_t i, size_t j) {
	size_t new_rdim = static_cast<size_t>(rdim / 2);
	size_t new_cdim = static_cast<size_t>(cdim / 2);
	return new SplittableMatrix<T>(
		root,
		rDis + (i * new_rdim),
		cDis + (j * new_cdim),
		new_rdim,
		new_cdim
	);
}

template<class T>
inline size_t SplittableMatrix<T>::map2Dto1DIndex(size_t i, size_t j) const {
	return (i + rDis) * root->cdim + (j + cDis);
}

template<class T>
inline T SplittableMatrix<T>::get(size_t index) const {
	return root->data[index];
}

template<class T>
inline void SplittableMatrix<T>::set(size_t index, T value) {
	root->data[index] = value;
}

template<class T>
inline void SplittableMatrix<T>::print() const {
	for (size_t i = 0; i < rdim; i++) {
		for (size_t j = 0; j < cdim; j++) {
			printf("%5.2f ", get(i, j));
		}
		printf("\n");
	}
}

template<class T>
inline bool SplittableMatrix<T>::is_same(SplittableMatrix<T>& lhs, SplittableMatrix<T>& rhs) {
	if (lhs.rdim != rhs.rdim || lhs.cdim != rhs.cdim) return false;
	return Buffer<T>::is_same(*(lhs.root), *(rhs.root));
}

#endif // MATRIX_VIEW_SPLITTABLE_H
