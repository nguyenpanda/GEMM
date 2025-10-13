#ifndef _MATRIX__CORE_MACRO_H
#define _MATRIX__CORE_MACRO_H

#if OMP_ENABLE == 1 && !defined(_OPENMP)
#error "OMP_ENABLE = 1 but can't find _OPENMP. Add OpenMP flags to compiler"
#endif // #if OMP_ENABLE == 0 || !defined(_OPENMP)

#if OMP_ENABLE == 1 && defined(_OPENMP)
	#include <omp.h>
#endif // OMP_ENABLE == 1 && defined(_OPENMP)

#if OMP_ENABLE == 0
	#undef _OPENMP
	#define omp_get_max_threads() 1
    #define omp_get_num_procs() 1
#endif // OMP_ENABLE == 0

#endif // _MATRIX__CORE_MACRO_H
