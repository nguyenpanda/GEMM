#include "gemm.h"

int main() {
	printf(GREEN "Starting the program\n" RESET);

	#pragma omp parallel
	printf(CYAN "[Thread=%d] " RESET "Hello, World!\n", omp_get_thread_num());

	printf(GREEN "Ending the program\n" RESET);
	return 0;
}
