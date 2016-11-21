
#include "histogram_atomics.cuh"
#include <stdio.h>

__global__ void histogram_gmem_atomics(const IN_TYPE *in, int width, int height, unsigned int *out)
{
    // pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // grid dimensions
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    // linear thread index within 2D block
    int t = threadIdx.x + threadIdx.y * blockDim.x;

    // total threads in 2D block
    int nt = blockDim.x * blockDim.y;

    // linear block index within 2D grid
    int g = blockIdx.x + blockIdx.y * gridDim.x;

    // initialize temporary accumulation array in global memory
    unsigned int *gmem = out + g * NUM_PARTS;
    for (int i = t; i < 3 * NUM_BINS; i += nt) gmem[i] = 0;

    // process pixels
    // updates our block's partial histogram in global memory
    for (int col = x; col < width; col += nx)
        for (int row = y; row < height; row += ny) {
            unsigned int r = (unsigned int)(256 * in[row * width + col].x);
            unsigned int g = (unsigned int)(256 * in[row * width + col].y);
            unsigned int b = (unsigned int)(256 * in[row * width + col].z);
            atomicAdd(&gmem[NUM_BINS * 0 + r], 1);
            atomicAdd(&gmem[NUM_BINS * 1 + g], 1);
            atomicAdd(&gmem[NUM_BINS * 2 + b], 1);
    }
}

__global__ void histogram_smem_atomics(const IN_TYPE *in, int width, int height, unsigned int *out)
{
    // pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // grid dimensions
    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;

    // linear thread index within 2D block
    int t = threadIdx.x + threadIdx.y * blockDim.x;

    // total threads in 2D block
    int nt = blockDim.x * blockDim.y;

    // linear block index within 2D grid
    int g = blockIdx.x + blockIdx.y * gridDim.x;

    // initialize temporary accumulation array in shared memory
    __shared__ unsigned int smem[3 * NUM_BINS + 3];
    for (int i = t; i < 3 * NUM_BINS + 3; i += nt) smem[i] = 0;
    __syncthreads();

    // process pixels
    // updates our block's partial histogram in shared memory
    for (int col = x; col < width; col += nx)
        for (int row = y; row < height; row += ny) {
            unsigned int r = (unsigned int)(256 * in[row * width + col].x);
            unsigned int g = (unsigned int)(256 * in[row * width + col].y);
            unsigned int b = (unsigned int)(256 * in[row * width + col].z);
            atomicAdd(&smem[NUM_BINS * 0 + r + 0], 1);
            atomicAdd(&smem[NUM_BINS * 1 + g + 1], 1);
            atomicAdd(&smem[NUM_BINS * 2 + b + 2], 1);
        }
        __syncthreads();

        // write partial histogram into the global memory
        out += g * NUM_PARTS;
        for (int i = t; i < NUM_BINS; i += nt) {
        out[i + NUM_BINS * 0] = smem[i + NUM_BINS * 0];
        out[i + NUM_BINS * 1] = smem[i + NUM_BINS * 1 + 1];
        out[i + NUM_BINS * 2] = smem[i + NUM_BINS * 2 + 2];
    }
}

__global__ void histogram_final_accum(const unsigned int *in, int n, unsigned int *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 3 * NUM_BINS) {
        unsigned int total = 0;
        for (int j = 0; j < n; j++)
            total += in[i + NUM_PARTS * j];
        out[i] = total;
    }
}
