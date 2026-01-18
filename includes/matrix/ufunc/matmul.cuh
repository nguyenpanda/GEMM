#ifndef MATRIX_UFUNC_CUDA_H
#define MATRIX_UFUNC_CUDA_H

#include "../viewer.h"
#include <assert.h>

namespace cuda {

template<class T>
__global__ void tited_kernel(T* out, T* lhs, T* rhs, size_t N) {	
	assert(CUDA_TILE_WIDTH == blockDim.x);
	
	int by = blockIdx.y;
	int bx = blockIdx.x;

	int ty = threadIdx.y;
	int tx = threadIdx.x;

	int i = blockDim.y * by + ty;
	int j = blockDim.x * bx + tx;

	__shared__ T sh_lhs[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
	__shared__ T sh_rhs[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];

	T result = 0;
	for (int phase = 0; phase < N / CUDA_TILE_WIDTH; phase++) {
		sh_lhs[ty][tx] = lhs[i * N + phase * CUDA_TILE_WIDTH + tx];
		sh_rhs[ty][tx] = rhs[(phase * CUDA_TILE_WIDTH + ty) * N + j];
		__syncthreads();

		for (int k = 0; k < CUDA_TILE_WIDTH; k++) {
			result += sh_lhs[ty][k] + sh_rhs[k][tx];
		}
		__syncthreads();
	}

	out[i * N + j] = result;
}

}; // namespace cuda


namespace ufunc {
namespace matmul {

template<class T>
class Tited {
public:
    __host__ static inline void operate(Buffer<T>& out, const Buffer<T>& lhs, const Buffer<T>& rhs) {
        const int TILE = CUDA_TILE_WIDTH;
        const int NS = omp_get_num_procs();

        size_t size_lhs = lhs.rdim * lhs.cdim * sizeof(T);
        size_t size_rhs = rhs.rdim * rhs.cdim * sizeof(T);
        size_t size_out = out.rdim * out.cdim * sizeof(T);

        T *gpu_lhs, *gpu_rhs, *gpu_out;
        CUDA_CHECK(cudaMalloc(&gpu_lhs, size_lhs));
        CUDA_CHECK(cudaMalloc(&gpu_rhs, size_rhs));
        CUDA_CHECK(cudaMalloc(&gpu_out, size_out));

        cudaStream_t streams[NS];
        for (int i = 0; i < NS; i++)
            CUDA_CHECK(cudaStreamCreate(&streams[i]));

        size_t chunk_lhs = size_lhs / NS;
        size_t chunk_rhs = size_rhs / NS;

        for (int i = 0; i < NS; i++) {
            size_t offL = i * chunk_lhs;
            size_t offR = i * chunk_rhs;

            cudaMemcpyAsync(
                gpu_lhs + offL/sizeof(T),
                lhs.data + offL/sizeof(T),
                (i == NS - 1 ? size_lhs - offL : chunk_lhs),
                cudaMemcpyHostToDevice,
                streams[i]
            );

            cudaMemcpyAsync(
                gpu_rhs + offR/sizeof(T),
                rhs.data + offR/sizeof(T),
                (i == NS-1 ? size_rhs - offR : chunk_rhs),
                cudaMemcpyHostToDevice,
                streams[i]
            );
        }

        cudaStream_t computeStream;
        CUDA_CHECK(cudaStreamCreate(&computeStream));

        cudaEvent_t copyDone;
        CUDA_CHECK(cudaEventCreate(&copyDone));

        for (int i = 0; i < NS; i++)
            cudaEventRecord(copyDone, streams[i]);

        cudaStreamWaitEvent(computeStream, copyDone, 0);

        dim3 block(TILE, TILE);
        dim3 grid(
            (out.cdim + TILE - 1) / TILE,
            (out.rdim + TILE - 1) / TILE
        );

        cuda::tited_kernel<T><<<grid, block, 0, computeStream>>>(
            gpu_out, gpu_lhs, gpu_rhs, out.rdim
        );

        cudaMemcpyAsync(
            out.data, gpu_out, size_out,
            cudaMemcpyDeviceToHost, computeStream
        );

        cudaStreamSynchronize(computeStream);

        for (int i = 0; i < NS; i++)
            cudaStreamDestroy(streams[i]);

        CUDA_CHECK(cudaStreamDestroy(computeStream));
        CUDA_CHECK(cudaEventDestroy(copyDone));

        CUDA_CHECK(cudaFree(gpu_lhs));
        CUDA_CHECK(cudaFree(gpu_rhs));
        CUDA_CHECK(cudaFree(gpu_out));
    }
};


}; // namespace addition
}; // namespace ufunc

#endif // MATRIX_UFUNC_CUDA_H
