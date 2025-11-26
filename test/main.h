#include <stdlib.h>
#include <stdio.h>
#include <gemm.h>

static size_t MATRIX_RDIM;
static size_t MATRIX_CDIM;
static size_t ADD_RECURSIVE_THRESHOLD;
static size_t MUL_RECURSIVE_THRESHOLD;

template<class T>
int INIT_METADATA() {
	printf("OMP_NUM_THREADS = %d\n", omp_get_max_threads());

	MATRIX_RDIM = static_cast<size_t>(atoi(getenv("MATRIX_RDIM")));
	MATRIX_CDIM = static_cast<size_t>(atoi(getenv("MATRIX_CDIM")));
	printf("MATRIX_RDIM = %zu\n", MATRIX_RDIM);
	printf("MATRIX_CDIM = %zu\n", MATRIX_CDIM);

	ADD_RECURSIVE_THRESHOLD = static_cast<size_t>(atoi(getenv("ADD_RECURSIVE_THRESHOLD")));
	ufunc::addition::OmpForkJoin<T>::set_threshold(ADD_RECURSIVE_THRESHOLD);
	printf("ADD_RECURSIVE_THRESHOLD = %zu\n", ADD_RECURSIVE_THRESHOLD);

	MUL_RECURSIVE_THRESHOLD = static_cast<size_t>(atoi(getenv("MUL_RECURSIVE_THRESHOLD")));
	ufunc::matmul::OmpForkJoin<T>::set_threshold(MUL_RECURSIVE_THRESHOLD);
	ufunc::matmul::OmpStrassen<float>::set_threshold(MUL_RECURSIVE_THRESHOLD);
	printf("MUL_RECURSIVE_THRESHOLD = %zu\n", MUL_RECURSIVE_THRESHOLD);

	printf("PRECISION_MODE = %s, MODE = %d\n", PRECISION_STATUS, PRECISION_MODE);
	printf("MATMUL_ORDER = %s\n", MATMUL_ORDER == MATMUL_IJK ? "IJK" : "IKJ");

	return 0;
}
