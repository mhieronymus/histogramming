
//#include "histogram_atomics.cuh"
#include <stdio.h>
#include <float.h>
//#include "device_functions.h"

#define %(c_precision_def)s
#define fType %(c_ftype)s
#define iType %(c_itype)s
#define uiType %(c_uitype)s
#define N_FLAT_BINS %(n_flat_bins)i
// See ieee floating point specification
#define CUDART_INF_F __ull_as_fType(0x7ff0000000000000ULL)
#define CUDART_NEG_INF_F __ull_as_fType(0xfff0000000000000ULL)

extern __shared__ fType shared[];

__device__ fType __ull_as_fType(unsigned long long int a)
{
    union {unsigned long long a; fType b;} u;
    u.a = a;
    return u.b;
}

__device__ unsigned long long int __fType_as_ull(fType a)
{
    union {fType a; unsigned long long int b;} u;
    u.a = a;
    return u.b;
}

// You can use atomicCAS() to create an atomicMax for any type.
// See https://docs.nvidia.com/cuda/cuda-c-programming-guide/ for further
// information.
__device__ fType atomicMaxfType(fType *address, fType val)
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

__device__ fType atomicMinfType(fType *address, fType val)
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

__device__ void max_min_reduce(const fType *d_array, const iType n_elements,
    fType *d_max, fType *d_min)
{
    // First n_elements entries are used for max reduction the last
    // n_elements entries are used for min reduction.
    fType *shared_max = (fType*)shared;
    fType *shared_min = (fType*)&shared[blockDim.x];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    // One could as well initialize the array with values from d_array.
    if(gid < n_elements)
    {
        shared_max[tid] = CUDART_NEG_INF_F;
        shared_min[tid] = CUDART_INF_F;
    }
    // if(gid == 0)
         // printf("shared_max %%f, shared_min %%f by thread %%d \n", shared_max[0], shared_min[0], gid);

    // Start max reduction in each block
    while(gid < n_elements)
    {
        shared_max[tid] = max(shared_max[tid], d_array[gid]);
        shared_min[tid] = min(shared_min[tid], d_array[gid]);
        gid += gridDim.x * blockDim.x;
    }
    __syncthreads();

    gid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i=blockDim.x/2; i > 0; i >>= 1)
    {
        // First check: For reduce algorithm
        // Second check: Do not access memory outside of our input elements
        // Third check: If there are less elements than threads in one block
        // do not access out of bounds. Only for "last block" and
        // n_elements/blockDim.x != 0
        if(tid < i && gid < n_elements && i < n_elements)
        {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + i]);
            shared_min[tid] = min(shared_min[tid], shared_min[tid + i]);
            // if(gid == 0)
            //     printf("shared_max %%f, shared_min %%f by thread %%d \n", shared_max[tid], shared_min[tid], gid);
        }
        __syncthreads();
    }
    // if(gid == 0)
    //     printf("shared_max %%f, shared_min %%f by thread %%d \n", shared_max[0], shared_min[0], gid);
    // Now return max value of all blocks in global memory
    if(tid == 0)
    {
        d_max[0] = CUDART_NEG_INF_F;
        d_min[0] = CUDART_INF_F;
        atomicMaxfType(d_max, shared_max[0]);
        atomicMinfType(d_min, shared_min[0]);
    }
}

// TODO: Make different methods with bins = int, array, (int, int), multiple arrays, combination of both
__global__ void histogram_gmem_atomics(const fType *in,  const iType length,
        const iType no_of_dimensions,  const iType no_of_bins,
        uiType *out, fType *max_in, fType *min_in)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    // total threads
    unsigned int nt = blockDim.x * gridDim.x;

    // initialize temporary histogram for each block in global memory
    uiType *gmem = out + (no_of_bins * no_of_dimensions) * blockIdx.x;
    // Each thread writes zeros to global memory
    for(unsigned int i = tid; i < no_of_bins * no_of_dimensions;
            i += blockDim.x)
    {
        gmem[i] = 0;
    }

    // Find min and max value in each dimension and store those in local memory
    unsigned int dimension_length = length/no_of_dimensions;
    for(unsigned int i = 0; i<no_of_dimensions; i++)
    {
        max_min_reduce(&in[i*dimension_length], dimension_length, &max_in[i], &min_in[i]);
        __syncthreads();
    }

    // Process input data by updating the histogram of each block in global
    // memory.
    for(unsigned int i = gid * no_of_dimensions; i < length;
            i += no_of_dimensions * nt)
    {
        int current_bin = 0;
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
            // Comparing floats/doubles like that looks suspicious...
            if(val == max_in[d])
                current_bin--;
            current_bin += ((val-min_in[d])/bin_width ) * power_bins;
            // if((gid > 255 && gid < length) || gid == 0){
            //     printf("Current_bin %%d by thread %%u with  value %%f \n", current_bin, gid, val);
            //    printf("local_min[d] %%f, local_max[d] %%f by thread %%u with bin_width %%f and val/bin_width %%f \n", min_in[d], max_in[d], gid, bin_width, val/bin_width);
            //}
        }
        // Avoid illegal memory access
        if(current_bin < no_of_bins * no_of_dimensions)
        {
            // if((gid > 255 && gid < length) || gid == 0)
            //     printf("Current_bin %%d by thread %%u with   \n", current_bin, gid);
            atomicAdd(&gmem[current_bin], 1);
        }
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

__global__ void histogram_final_accum(const uiType *in,
        iType no_of_histograms, uiType *out,
        iType no_of_bins, iType no_of_dimensions)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int nx = blockDim.x * gridDim.x;
    iType histo_length = no_of_bins * no_of_dimensions;
    // Each thread merges values for another bin
    for(unsigned int current_bin = gid; current_bin < no_of_bins;
            current_bin += nx)
    {
        if(current_bin < no_of_bins)
        {
            uiType total = 0;
            for(unsigned int j = 0; j < no_of_histograms; j++)
            {
                total += in[histo_length * j + current_bin];
            }
            out[current_bin] = total;
        }
    }
}
