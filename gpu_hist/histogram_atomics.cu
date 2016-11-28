
#include "histogram_atomics.cuh"
#include <stdio.h>


// You can use atomicCAS() to create an atomicMax for any type.
// See https://docs.nvidia.com/cuda/cuda-c-programming-guide/ for further
// information.
__device__ fType atomicMaxAny(fType *adress, fType val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    while(val > __ull_as_fType(old))
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __fType_as_ull(val));
    }
    return __ull_as_fType(old);
}

__device__ fType atomicMinAny(fType *adress, fType val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    while(val < __ull_as_fType(old))
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __fType_as_ull(val));
    }
    return __ull_as_fType(old);
}

__global__ void max_min_reduce(const fType *d_array, const size_t length,
    fType d_max, fType d_min)
{
    extern __shared__ fType shared_max[], shared_min[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // This is necessary for max(...) in the next loops.
    if(gid < length)
    {
        shared_max[tid] = d_array[gid];
        shared_min[tid] = d_array[gid];
        gid += gridDim.x * blockDim.x;
    }
    // Start max reduction in each block
    while(gid < length)
    {
        shared_max[tid] = max(shared[tid], d_array[gid]);
        shared_min[tid] = min(shared[tid], d_array[gid]);
        gid += gridDim.x * blockDim.x;
    }
    __syncthreads();

    gid = blockIdx.x * blockDim.x + threadIdx.x;
    for(unsigned int i=blockDim.x/2; i > 0; i >>= 1)
    {
        if(tid < i && gid < elements)
        {
            shared_max[tid] = max(shared[tid], shared[tid + i]);
            shared_min[tid] = min(shared[tid], shared[tid + i]);
        }
        __syncthreads();
    }

    // Now return max value of all blocks
    if(tid == 0)
    {
        atomicMaxAny(d_max, shared_max[0]);
        atomicMinAny(d_min, shared_min[0]);
    }
    __syncthreads();
}

// TODO: Make different methods with bins = int, array, (int, int), multiple arrays, combination of both
__global__ void histogram_gmem_atomics(const fType *in,  const size_t length,
        const size_t no_of_dimensions,  const size_t no_of_bins,
        unsigned int *out)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x

    // grid dimensions
    int nx = blockDim.x * gridDim.x;

    // total threads in block
    int nt = blockDim.x;

    // initialize temporary histogram for each block in global memory
    unsigned int *gmem = out + blockDim.x * blockIdx.x;
    // Each thread writes zeros to  global memory
    for(unsigned int i = gid; i < no_of_bins * no_of_dimensions;
            i += blockDim.x)
    {
        gmem[i] = 0;
    }

    // Find min and max value in each dimension.
    fType max_in[], min_in[]; // TODO: Check if __global__ is necessary.
    int dimension_length = length/no_of_dimensions;
    for(unsigned int i = 0; i<no_of_dimensions; i++)
    {
        max_min_reduce(in, dimension_length, max_in[i], min_in[i]);
    }

    // Process input data by updating the histogram of each block in global
    // memory.
    for(unsigned int i = threadId * no_of_dimensions; i < length;
            i += no_of_dimensions * nt)
    {
        unsigned int current_bin = 0;
        // Look at each dimension.
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType val = in[i+d];
            fType bin_width = (max_in[d]-min_in[d])/no_of_bins;
            int power_bins = 1;
            for(unsigned int k=0; k < d; k++)
            {
                power_bins = no_of_bins * power_bins;
            }
            current_bin += (val/bin_width) * power_bins;
        }
        atomicAdd(&gmem[current_bin], 1);
    }
}

// TODO: This function
// __global__ void histogram_smem_atomics(const fType *in, int width, int height, unsigned int *out)
// {
//     // pixel coordinates
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//     // grid dimensions
//     int nx = blockDim.x * gridDim.x;
//     int ny = blockDim.y * gridDim.y;
//
//     // linear thread index within 2D block
//     int t = threadIdx.x + threadIdx.y * blockDim.x;
//
//     // total threads in 2D block
//     int nt = blockDim.x * blockDim.y;
//
//     // linear block index within 2D grid
//     int g = blockIdx.x + blockIdx.y * gridDim.x;
//
//     // initialize temporary accumulation array in shared memory
//     __shared__ unsigned int smem[3 * NUM_BINS + 3];
//     for (int i = t; i < 3 * NUM_BINS + 3; i += nt) smem[i] = 0;
//     __syncthreads();
//
//     // process pixels
//     // updates our block's partial histogram in shared memory
//     for (int col = x; col < width; col += nx)
//         for (int row = y; row < height; row += ny) {
//             unsigned int r = (unsigned int)(256 * in[row * width + col].x);
//             unsigned int g = (unsigned int)(256 * in[row * width + col].y);
//             unsigned int b = (unsigned int)(256 * in[row * width + col].z);
//             atomicAdd(&smem[NUM_BINS * 0 + r + 0], 1);
//             atomicAdd(&smem[NUM_BINS * 1 + g + 1], 1);
//             atomicAdd(&smem[NUM_BINS * 2 + b + 2], 1);
//         }
//         __syncthreads();
//
//         // write partial histogram into the global memory
//         out += g * NUM_PARTS;
//         for (int i = t; i < NUM_BINS; i += nt) {
//         out[i + NUM_BINS * 0] = smem[i + NUM_BINS * 0];
//         out[i + NUM_BINS * 1] = smem[i + NUM_BINS * 1 + 1];
//         out[i + NUM_BINS * 2] = smem[i + NUM_BINS * 2 + 2];
//     }
// }

__global__ void histogram_final_accum(const unsigned int *in,
        int no_of_histograms, unsigned int *out,
        unsigned int no_of_bins, unsigned int no_of_dimensions)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int nx = blockDim.x * gridDim.x;
    unsigned int histo_length = no_of_bins * no_of_dimensions;
    // Each thread merges values for another bin
    for(unsigned int current_bin = gid; current_bin < no_of_bins;
            current_bin += nx)
    {
        if(current_bin < no_of_bins)
        {
            unsigned int total = 0;
            for(unsigned int j = 0; j < no_of_histograms; j++)
            {
                total += in[histo_length * j + current_bin];
            }
            out[current_bin] = total;
        }
    }
}
