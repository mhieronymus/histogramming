
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

__global__ void max_min_reduce(const fType *d_array, const iType n_elements,
    const iType no_of_dimensions, fType *d_max, fType *d_min)
{
    // First n_elements entries are used for max reduction the last
    // n_elements entries are used for min reduction.
    fType *shared_max = (fType*)shared;
    fType *shared_min = (fType*)&shared[blockDim.x];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Init global max and min value. This is a separated loop to avoid
    // race conditions.
    for(int d = 0; d < no_of_dimensions; d++)
    {
        if(gid == 0)
        {
            d_max[d] = CUDART_NEG_INF_F;
            d_min[d] = CUDART_INF_F;
        }
    }

    // Max- and Min-Reduce for each dimension
    for(int d = 0; d < no_of_dimensions; d++)
    {
        // Initialize shared memory with input memory
        if(gid < n_elements)
        {
            shared_max[tid] = d_array[gid*no_of_dimensions+d];
            shared_min[tid] = d_array[gid*no_of_dimensions+d];
            gid += gridDim.x * blockDim.x;
        }

        // Start max reduction in each block with left overs from input array.
        // If there are more elements than threads, then we copy the next
        // elements from input if they are bigger/lower than the last copied
        // values.
        while(gid < n_elements && gid >= n_elements)
        {
            shared_max[tid] = max(shared_max[tid],
                d_array[gid*no_of_dimensions+d]);
            shared_min[tid] = min(shared_min[tid],
                d_array[gid*no_of_dimensions+d]);
            gid += gridDim.x * blockDim.x;
        }
        __syncthreads();
        gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Blockwise reduction
        for(int i=blockDim.x/2; i > 0; i >>= 1)
        {
            // First check: For reduce algorithm
            // Second check: Do not access memory outside of our input elements
            // Third check: If there are less elements than threads in one block
            // do not access out of bounds. Only for "last block" and
            // n_elements/blockDim.x != 0
            if(tid < i && gid < n_elements && gid + i < n_elements)
            {
                shared_max[tid] = max(shared_max[tid], shared_max[tid + i]);
                shared_min[tid] = min(shared_min[tid], shared_min[tid + i]);
            }
            __syncthreads();
        }
        // Now return max value of all blocks in global memory
        if(tid == 0 && gid < n_elements)
        {
            atomicMaxfType(&d_max[d], shared_max[0]);
            atomicMinfType(&d_min[d], shared_min[0]);
        }
        // printf("Global: max %%f, min %%f by thread %%d \n", d_max[d], d_min[d], gid);
        // I don't think a syncthreads is needed here.
        // __syncthreads();
    }
}

// Takes max and min value for each dimension and the number of bins and
// returns a histogram with equally sized bins.
__global__ void histogram_gmem_atomics(const fType *in,  const iType length,
        const iType no_of_dimensions,  const iType no_of_bins,
        const iType no_of_flat_bins, uiType *out, fType *max_in, fType *min_in)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    // initialize temporary histogram for each block in global memory
    uiType *gmem = out + no_of_flat_bins * blockIdx.x;
    // Each thread writes zeros to global memory
    for(unsigned int i = tid; i < no_of_flat_bins;
            i += blockDim.x)
    {
        gmem[i] = 0;
    }

    // Process input data by updating the histogram of each block in global
    // memory. Each thread processes one element with all its dimensions at a
    // time.
    for(unsigned int i = gid*no_of_dimensions; i < length;
        i+=no_of_dimensions*total_threads)
    {
        int current_bin = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            // if(gid == 0)
                // printf("current_bin %%d\n", current_bin);
            fType bin_width = (max_in[d]-min_in[d])/no_of_bins;
            fType val = in[i + d];
            // Get the bin in the current dimension
            int tmp_bin = (val-min_in[d])/bin_width;
            if(tmp_bin >= no_of_bins) tmp_bin--;
            // Get the right place in the histogram
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins * power_bins;
            }
            // if(gid == 0)
            // {
                //  printf("B val %%f, d %%d, current_bin %%d, by thread %%d \n", val, d, current_bin, gid);
                //  printf("B power_bins %%d, tmp_bin %%d, bin_width %%f by thread %%d \n", power_bins, tmp_bin, bin_width, gid);

            //  }
            current_bin += tmp_bin * power_bins;
            // if(gid == 0)
            // {
                //  printf("val %%f, d %%d, current_bin %%d, by thread %%d \n", val, d, current_bin, gid);
                //  printf("power_bins %%d, tmp_bin %%d, bin_width %%f by thread %%d \n", power_bins, tmp_bin, bin_width, gid);
            //  }
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAdd(&gmem[current_bin], 1);
        }
    }

    // // Process input data by updating the histogram of each block in global
    // // memory.
    // for(unsigned int d = 0; d < no_of_dimensions; d++)
    // {
    //     fType bin_width = (max_in[d]-min_in[d])/no_of_bins;
    //     int power_bins = 1;
    //     for(unsigned int k=0; k < d; k++)
    //     {
    //         power_bins = no_of_bins * power_bins;
    //     }
    //
    //     for(unsigned int i = gid * no_of_dimensions+d; i < length;
    //             i += no_of_dimensions * total_threads)
    //     {
    //         int current_bin = 0;
    //         fType val = in[i];
    //         int tmp_bin = (val-min_in[d])/bin_width;
    //         if(tmp_bin >= no_of_bins) tmp_bin--;
    //         current_bin += tmp_bin * power_bins;
    //
    //         // Avoid illegal memory access
    //         if(current_bin < no_of_flat_bins)
    //         {
    //             atomicAdd(&gmem[current_bin], 1);
    //         }
    //     }
    // }
}

// Takes edges for each dimension and the number of bins and
// returns a histogram with equally sized bins.
__global__ void histogram_gmem_atomics_with_edges(const fType *in,  const iType length,
        const iType no_of_dimensions,  const iType no_of_bins,
        uiType *out, fType *edges_in)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;

    // initialize temporary histogram for each block in global memory
    uiType *gmem = out + (no_of_bins * no_of_dimensions) * blockIdx.x;
    // Each thread writes zeros to global memory
    for(unsigned int i = tid; i < no_of_bins * no_of_dimensions;
            i += blockDim.x)
    {
        gmem[i] = 0;
    }

    // Process input data by updating the histogram of each block in global
    // memory.
    for(unsigned int i = gid * no_of_dimensions; i < length;
            i += no_of_dimensions * total_threads)
    {
        int current_bin = 0;
        // Look at each dimension.
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType val = in[i+d];
            while(val > edges_in[no_of_bins*d+current_bin+1])
            {
                current_bin++;

                if(current_bin > no_of_bins)
                {
                    // No bin available for this value
                    current_bin = no_of_bins * no_of_dimensions + 1;
                    break;
                }
            }
            int power_bins = 0;
            for(unsigned int k=0; k < d; k++)
            {
                power_bins = no_of_bins * power_bins;
            }
            current_bin += no_of_bins * power_bins;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_bins * no_of_dimensions)
        {
            atomicAdd(&gmem[current_bin], 1);
        }
    }
}

__global__ void histogram_smem_atomics_with_edges(const fType *in, const iType length,
        const iType no_of_dimensions,  const iType no_of_bins,
        uiType *out, fType *edges_in)
{
    // unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // unsigned int tid = threadIdx.x;
    // unsigned int total_threads = blockDim.x * gridDim.x;
    // unsigned int threads_per_block = blockDim.x;
    //
    // // initialize temporary accumulation array in shared memory
    // __shared__ uiType smem[no_of_bins * no_of_dimensions];
    // for(unsigned int i = gid; i < no_of_bins * no_of_dimensions;  i+= threads_per_block)
    // {
    //     smem[i] = 0;
    // }
    // __syncthreads();
    //
    // // Process input data by updating the histogram of each block in shared
    // // memory.
    // for(unsigned int i = gid * no_of_dimensions; i < length;
    //         i += no_of_dimensions * total_threads)
    // {
    //     int current_bin = 0;
    //     // Look at each dimension.
    //     for(unsigned int d = 0; d < no_of_dimensions; d++)
    //     {
    //         fType val = in[i+d];
    //         while(val > edges_in[no_of_bins*d+current_bin+1])
    //         {
    //             current_bin++;
    //
    //             if(current_bin > no_of_bins)
    //             {
    //                 // No bin available for this value
    //                 current_bin = no_of_bins * no_of_dimensions + 1;
    //                 break;
    //             }
    //         }
    //         int power_bins = 0;
    //         for(unsigned int k=0; k < d; k++)
    //         {
    //             power_bins = no_of_bins * power_bins;
    //         }
    //         current_bin += no_of_bins * power_bins;
    //     }
    //     // Avoid illegal memory access
    //     if(current_bin < no_of_bins * no_of_dimensions)
    //     {
    //         atomicAdd(&smem[current_bin], 1);
    //     }
    // }
    // __syncthreads();


    //
    // // pixel coordinates
    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;
    //
    // // grid dimensions
    // int nx = blockDim.x * gridDim.x;
    // int ny = blockDim.y * gridDim.y;
    //
    // // linear thread index within 2D block
    // int t = threadIdx.x + threadIdx.y * blockDim.x;
    //
    // // total threads in 2D block
    // int nt = blockDim.x * blockDim.y;
    //
    // // linear block index within 2D grid
    // int g = blockIdx.x + blockIdx.y * gridDim.x;
    //
    // // initialize temporary accumulation array in shared memory
    // __shared__ unsigned int smem[3 * NUM_BINS + 3];
    // for (int i = t; i < 3 * NUM_BINS + 3; i += nt) smem[i] = 0;
    // __syncthreads();
    //
    // // process pixels
    // // updates our block's partial histogram in shared memory
    // for (int col = x; col < width; col += nx)
    //     for (int row = y; row < height; row += ny) {
    //         unsigned int r = (unsigned int)(256 * in[row * width + col].x);
    //         unsigned int g = (unsigned int)(256 * in[row * width + col].y);
    //         unsigned int b = (unsigned int)(256 * in[row * width + col].z);
    //         atomicAdd(&smem[NUM_BINS * 0 + r + 0], 1);
    //         atomicAdd(&smem[NUM_BINS * 1 + g + 1], 1);
    //         atomicAdd(&smem[NUM_BINS * 2 + b + 2], 1);
    //     }
    // __syncthreads();
    //
    //     // write partial histogram into the global memory
    // out += g * NUM_PARTS;
    // for (int i = t; i < NUM_BINS; i += nt) {
    //     out[i + NUM_BINS * 0] = smem[i + NUM_BINS * 0];
    //     out[i + NUM_BINS * 1] = smem[i + NUM_BINS * 1 + 1];
    //     out[i + NUM_BINS * 2] = smem[i + NUM_BINS * 2 + 2];
    // }
}


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
        iType no_of_bins, iType histo_length, iType no_of_dimensions)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    // Each thread merges values for another bin
    for(unsigned int current_bin = gid; current_bin < histo_length;
            current_bin += total_threads)
    {
        if(current_bin < histo_length)
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
