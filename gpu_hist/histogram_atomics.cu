#include <stdio.h>
#include <float.h>

#define %(c_precision_def)s
#define fType %(c_ftype)s
#define iType %(c_itype)s
#define histoType %(c_histotype)s
#define changeType %(c_changetype)s

// See ieee floating point specification
#define CUDART_INF_F __ull_as_fType(0x7ff0000000000000ULL)
#define CUDART_NEG_INF_F __ull_as_fType(0xfff0000000000000ULL)

// This function is used to convert inf and -inf from ull to ftype.
// Since ftype is either float32 or float64, there are two ways.
// float64:
// Simple converting
// float32:
// Converting leads to overflow -> value will be 0. Get the correct sign
// and return inf.
__device__ fType __ull_as_fType(unsigned long long int a)
{
    union {unsigned long long int a; fType b;} u;
    u.a = a;
    fType f = u.b;
    if(f != 0) return f;
    // float32
    if(a == 0x7ff0000000000000ULL)
    {
        int inf = 0x7F800000;
        return *(fType*)&inf;
    } else
    {
        // -inf
        int neg_inf = 0xff800000;
        return *(fType*)&neg_inf;
    }
}

__device__ fType __change_as_fType(changeType a)
{
    union {changeType a; fType b;} u;
    u.a = a;
    return u.b;
}

__device__ unsigned long long int __fType_as_change(fType a)
{
    union {fType a; changeType b;} u;
    u.a = a;
    return u.b;
}

// You can use atomicCAS() to create an atomicMax for any type.
// See https://docs.nvidia.com/cuda/cuda-c-programming-guide/ for further
// information.
__device__ fType atomicMaxfType(fType *address, fType val)
{
    changeType* address_as_ull = (changeType*) address;
    changeType old = *address_as_ull, assumed;
    while(val > __change_as_fType(old))
    {
         assumed = old;
         old = atomicCAS(address_as_ull, assumed, __fType_as_change(val));
    }
    return __change_as_fType(old);
}

__device__ fType atomicMinfType(fType *address, fType val)
{
    changeType* address_as_ull = (changeType*) address;
    changeType old = *address_as_ull, assumed;
    while(val < __change_as_fType(old))
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __fType_as_change(val));
    }
    return __change_as_fType(old);
}

// This method is used when weights are given. The histogram should have the
// same type as fType.
__device__ fType atomicAddfType(fType *address, fType val)
{
    changeType* address_as_ull = (changeType*) address;
    changeType old = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __fType_as_change(val + __change_as_fType(assumed)));
    } while(assumed != old);
    return __change_as_fType(old);
}

__global__ void max_min_reduce(const fType *d_array, const iType n_elements,
    const iType no_of_dimensions, fType *d_max, fType *d_min)
{
    // First n_elements entries are used for max reduction the last
    // n_elements entries are used for min reduction.
    extern __shared__ fType shared[];
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
        while(gid < n_elements)
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
            // Second check: If there are less elements than threads in one block
            // do not access out of bounds. Only for "last block" and
            // n_elements/blockDim.x != 0
            if(tid < i && gid + i < n_elements)
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
    }
}

// Takes max and min value for each dimension and the number of bins and
// returns a histogram with equally sized bins.
__global__ void histogram_gmem_atomics(const fType *in,  const iType length,
        const iType no_of_dimensions,  const iType *no_of_bins,
        const iType no_of_flat_bins, histoType *out, fType *max_in, fType *min_in)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;

    // initialize temporary histogram for each block in global memory
    histoType *gmem = out + no_of_flat_bins * blockIdx.x;
    // Each thread writes zeros to global memory
    for(unsigned int i = tid; i < no_of_flat_bins; i += blockDim.x)
    {
        gmem[i] = 0;
    }
    __syncthreads();
    // Process input data by updating the histogram of each block in global
    // memory. Each thread processes one element with all its dimensions at a
    // time.
    for(unsigned int i = gid*no_of_dimensions; i < length;
        i+=no_of_dimensions*total_threads)
    {
        int current_bin = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType bin_width = (max_in[d]-min_in[d])/no_of_bins[d];
            fType val = in[i + d];
            // Get the bin in the current dimension
            if(val < min_in[d] || val > max_in[d])
            {
                current_bin = no_of_flat_bins+1;
                break;
            }
            int tmp_bin = (val-min_in[d])/bin_width;
            if(tmp_bin >= no_of_bins[d]) tmp_bin--;
            // Get the right place in the histogram
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAdd(&gmem[current_bin], 1);
        }
    }
}

// Takes edges for each dimension and the number of bins and
// returns a histogram with equally sized bins.
__global__ void histogram_gmem_atomics_with_edges(const fType *in,
        const iType length, const iType no_of_dimensions,
        const iType *no_of_bins, const iType no_of_flat_bins,
        histoType *out, const fType *edges_in)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    unsigned int threads_per_block = blockDim.x;

    // initialize temporary histogram for each block in global memory
    histoType *gmem = out + no_of_flat_bins * blockIdx.x;
    // Each thread writes zeros to global memory
    for(unsigned int i = tid; i < no_of_flat_bins; i += threads_per_block)
    {
        gmem[i] = 0;
    }
    __syncthreads();

    // Process input data by updating the histogram of each block in global
    // memory.
    for(unsigned int i = gid*no_of_dimensions; i < length;
            i += no_of_dimensions*total_threads)
    {
        int current_bin = 0;
        int bins_offset = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType val = in[i + d];
            int tmp_bin = 0;
            while(val > edges_in[bins_offset+tmp_bin+1]
                && tmp_bin < no_of_bins[d]-1)
            {
                 tmp_bin++;
            }
            if(val > edges_in[bins_offset+tmp_bin+1] || val < edges_in[bins_offset])
            {
                bins_offset += no_of_bins[d]+1;
                current_bin = no_of_flat_bins + 1;
                break;
            }
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
            bins_offset += no_of_bins[d]+1;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAdd(&gmem[current_bin], 1);
        }
    }
}

// Takes max and min value for each dimension and the number of bins and
// returns a histogram with equally sized bins.
__global__ void histogram_gmem_atomics_weights(const fType *in,  const iType length,
        const iType no_of_dimensions,  const iType *no_of_bins,
        const iType no_of_flat_bins, fType *out, fType *max_in, fType *min_in,
        const fType *weights)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;

    // initialize temporary histogram for each block in global memory
    fType *gmem = out + no_of_flat_bins * blockIdx.x;
    // Each thread writes zeros to global memory
    for(unsigned int i = tid; i < no_of_flat_bins; i += blockDim.x)
    {
        gmem[i] = 0;
    }
    __syncthreads();
    // Process input data by updating the histogram of each block in global
    // memory. Each thread processes one element with all its dimensions at a
    // time.
    for(unsigned int i = gid*no_of_dimensions; i < length;
        i+=no_of_dimensions*total_threads)
    {
        int current_bin = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType bin_width = (max_in[d]-min_in[d])/no_of_bins[d];
            fType val = in[i + d];
            // Get the bin in the current dimension
            if(val < min_in[d] || val > max_in[d])
            {
                current_bin = no_of_flat_bins+1;
                break;
            }
            int tmp_bin = (val-min_in[d])/bin_width;
            if(tmp_bin >= no_of_bins[d]) tmp_bin--;
            // Get the right place in the histogram
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAddfType(&gmem[current_bin], weights[i/no_of_dimensions]);
        }
    }
}

// Takes edges for each dimension and the number of bins and
// returns a histogram with equally sized bins.
__global__ void histogram_gmem_atomics_with_edges_weights(const fType *in,
        const iType length, const iType no_of_dimensions,
        const iType *no_of_bins, const iType no_of_flat_bins,
        fType *out, const fType *edges_in, const fType *weights)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    unsigned int threads_per_block = blockDim.x;

    // initialize temporary histogram for each block in global memory
    fType *gmem = out + no_of_flat_bins * blockIdx.x;
    // Each thread writes zeros to global memory
    for(unsigned int i = tid; i < no_of_flat_bins; i += threads_per_block)
    {
        gmem[i] = 0;
    }
    __syncthreads();

    // Process input data by updating the histogram of each block in global
    // memory.
    for(unsigned int i = gid*no_of_dimensions; i < length;
            i += no_of_dimensions*total_threads)
    {
        int current_bin = 0;
        int bins_offset = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType val = in[i + d];
            int tmp_bin = 0;
            while(val > edges_in[bins_offset+tmp_bin+1]
                && tmp_bin < no_of_bins[d]-1)
            {
                 tmp_bin++;
            }
            if(val > edges_in[bins_offset+tmp_bin+1] || val < edges_in[bins_offset])
            {
                bins_offset += no_of_bins[d]+1;
                current_bin = no_of_flat_bins + 1;
                break;
            }
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
            bins_offset += no_of_bins[d]+1;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAddfType(&gmem[current_bin], weights[i/no_of_dimensions]);
        }
    }
}

__global__ void histogram_smem_atomics(const fType *in,  const iType length,
        const iType no_of_dimensions,  const iType *no_of_bins,
        const iType no_of_flat_bins, histoType *out, fType *max_in, fType *min_in)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    unsigned int threads_per_block = blockDim.x;

    // initialize temporary accumulation array in shared memory
    extern __shared__ histoType smem2[];
    for(unsigned int i = tid; i < no_of_flat_bins;  i+= threads_per_block)
    {
        smem2[i] = 0;
    }
    __syncthreads();

    // Process input data by updating the histogram of each block in global
    // memory. Each thread processes one element with all its dimensions at a
    // time.
    for(unsigned int i = gid*no_of_dimensions; i < length;
        i+=no_of_dimensions*total_threads)
    {
        int current_bin = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType bin_width = (max_in[d]-min_in[d])/no_of_bins[d];
            fType val = in[i + d];
            // Get the bin in the current dimension
            if(val < min_in[d] || val > max_in[d])
            {
                current_bin = no_of_flat_bins+1;
                break;
            }
            int tmp_bin = (val-min_in[d])/bin_width;
            if(tmp_bin >= no_of_bins[d]) tmp_bin--;
            // Get the right place in the histogram
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAdd(&smem2[current_bin], 1);
        }
    }
    __syncthreads();
    // Write partial histograms in global memory
    out = &out[blockIdx.x * no_of_flat_bins];
    for(unsigned int i = tid; i < no_of_flat_bins;  i+= threads_per_block)
    {
        out[i] = smem2[i];
    }
}

__global__ void histogram_smem_atomics_with_edges(const fType *in,
        const iType length, const iType no_of_dimensions,
        const iType *no_of_bins, const iType no_of_flat_bins,
        histoType *out, const fType *edges_in)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    unsigned int threads_per_block = blockDim.x;

    // initialize temporary accumulation array in shared memory
    extern __shared__ histoType smem2[];
    for(unsigned int i = tid; i < no_of_flat_bins;  i += threads_per_block)
    {
        smem2[i] = 0;
    }
    __syncthreads();

    // Process input data by updating the histogram of each block in global
    // memory. Each thread processes one element with all its dimensions at a
    // time.
    for(unsigned int i = gid*no_of_dimensions; i < length;
        i += no_of_dimensions*total_threads)
    {
        int current_bin = 0;
        int bins_offset = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType val = in[i + d];
            int tmp_bin = 0;
            while(val > edges_in[bins_offset+tmp_bin+1]
                    && tmp_bin < no_of_bins[d]-1)
            {
                tmp_bin++;
            }
            if(val > edges_in[bins_offset+tmp_bin+1] || val < edges_in[bins_offset])
            {
                bins_offset += no_of_bins[d]+1;
                current_bin = no_of_flat_bins + 1;
                break;
            }
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
            bins_offset += no_of_bins[d]+1;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAdd(&smem2[current_bin], 1);
        }
    }
    __syncthreads();

    // Write partial histograms in global memory
    histoType *overall_out = &out[blockIdx.x * no_of_flat_bins];
    for(unsigned int i = tid; i < no_of_flat_bins;  i+= threads_per_block)
    {
        overall_out[i] = smem2[i];
    }
}

__global__ void histogram_smem_atomics_weights(const fType *in,  const iType length,
        const iType no_of_dimensions,  const iType *no_of_bins,
        const iType no_of_flat_bins, fType *out, fType *max_in, fType *min_in,
        const fType *weights)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    unsigned int threads_per_block = blockDim.x;

    // initialize temporary accumulation array in shared memory
    extern __shared__ fType smem[];
    for(unsigned int i = tid; i < no_of_flat_bins;  i+= threads_per_block)
    {
        smem[i] = 0;
    }
    __syncthreads();

    // Process input data by updating the histogram of each block in global
    // memory. Each thread processes one element with all its dimensions at a
    // time.
    for(unsigned int i = gid*no_of_dimensions; i < length;
        i+=no_of_dimensions*total_threads)
    {
        int current_bin = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType bin_width = (max_in[d]-min_in[d])/no_of_bins[d];
            fType val = in[i + d];
            // Get the bin in the current dimension
            if(val < min_in[d] || val > max_in[d])
            {
                current_bin = no_of_flat_bins+1;
                break;
            }
            int tmp_bin = (val-min_in[d])/bin_width;
            if(tmp_bin >= no_of_bins[d]) tmp_bin--;
            // Get the right place in the histogram
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
          atomicAddfType(&smem[current_bin], weights[i/no_of_dimensions]);
        }
    }
    __syncthreads();
    // Write partial histograms in global memory
    out = &out[blockIdx.x * no_of_flat_bins];
    for(unsigned int i = tid; i < no_of_flat_bins;  i+= threads_per_block)
    {
        out[i] = smem[i];
    }
}

__global__ void histogram_smem_atomics_with_edges_weights(const fType *in,
        const iType length, const iType no_of_dimensions,
        const iType *no_of_bins, const iType no_of_flat_bins,
        fType *out, const fType *edges_in, const fType *weights)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    unsigned int threads_per_block = blockDim.x;

    // initialize temporary accumulation array in shared memory
    extern __shared__ fType smem[];
    for(unsigned int i = tid; i < no_of_flat_bins;  i += threads_per_block)
    {
        smem[i] = 0;
    }
    __syncthreads();

    // Process input data by updating the histogram of each block in global
    // memory. Each thread processes one element with all its dimensions at a
    // time.
    for(unsigned int i = gid*no_of_dimensions; i < length;
        i += no_of_dimensions*total_threads)
    {
        int current_bin = 0;
        int bins_offset = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType val = in[i + d];
            int tmp_bin = 0;
            while(val > edges_in[bins_offset+tmp_bin+1]
                    && tmp_bin < no_of_bins[d]-1)
            {
                tmp_bin++;
            }
            if(val > edges_in[bins_offset+tmp_bin+1] || val < edges_in[bins_offset])
            {
                bins_offset += no_of_bins[d]+1;
                current_bin = no_of_flat_bins + 1;
                break;
            }
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
            bins_offset += no_of_bins[d]+1;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAddfType(&smem[current_bin], weights[i/no_of_dimensions]);
        }
    }
    __syncthreads();

    // Write partial histograms in global memory
    fType *overall_out = &out[blockIdx.x * no_of_flat_bins];
    for(unsigned int i = tid; i < no_of_flat_bins;  i+= threads_per_block)
    {
        overall_out[i] = smem[i];
    }
}

__global__ void histogram_final_accum(const histoType *in,
        const iType no_of_histograms, histoType *out, const iType histo_length)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    // Each thread merges values for another bin
    for(unsigned int current_bin = gid; current_bin < histo_length;
            current_bin += total_threads)
    {
        if(current_bin < histo_length)
        {
            histoType total = 0;
            for(unsigned int j = 0; j < no_of_histograms; j++)
            {
                total += in[histo_length * j + current_bin];
            }
            out[current_bin] = total;
        }
    }
}

__global__ void histogram_final_accum_weights(const fType *in,
        const iType no_of_histograms, fType *out, const iType histo_length)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    // Each thread merges values for another bin
    for(unsigned int current_bin = gid; current_bin < histo_length;
            current_bin += total_threads)
    {
        if(current_bin < histo_length)
        {
            fType total = 0;
            for(unsigned int j = 0; j < no_of_histograms; j++)
            {
                total += in[histo_length * j + current_bin];
            }
            out[current_bin] = total;
        }
    }
}

////////////////////// Following code does the same but uses different input
__global__ void max_min_reduce2(const fType *d_array_x, const iType n_elements,
    const fType *d_array_y, const fType *d_array_z,
    const iType no_of_dimensions, fType *d_max, fType *d_min)
{
    // First n_elements entries are used for max reduction the last
    // n_elements entries are used for min reduction.
    extern __shared__ fType shared[];
    fType *shared_max = (fType*)shared;
    fType *shared_min = (fType*)&shared[blockDim.x];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    const fType *d_array[3] = {d_array_x, d_array_y, d_array_z};

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
            shared_max[tid] = d_array[d][gid];
            shared_min[tid] = d_array[d][gid];
            gid += gridDim.x * blockDim.x;
        }

        // Start max reduction in each block with left overs from input array.
        // If there are more elements than threads, then we copy the next
        // elements from input if they are bigger/lower than the last copied
        // values.
        while(gid < n_elements)
        {
            shared_max[tid] = max(shared_max[tid], d_array[d][gid]);
            shared_min[tid] = min(shared_min[tid], d_array[d][gid]);
            gid += gridDim.x * blockDim.x;
        }
        __syncthreads();
        gid = blockIdx.x * blockDim.x + threadIdx.x;

        // Blockwise reduction
        for(int i=blockDim.x/2; i > 0; i >>= 1)
        {
            // First check: For reduce algorithm
            // Second check: If there are less elements than threads in one block
            // do not access out of bounds. Only for "last block" and
            // n_elements/blockDim.x != 0
            if(tid < i && gid + i < n_elements)
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
    }
}

// Takes max and min value for each dimension and the number of bins and
// returns a histogram with equally sized bins.
__global__ void histogram_gmem_atomics2(const fType *in_x,  const iType length,
        const fType *in_y, const fType *in_z,
        const iType no_of_dimensions,  const iType *no_of_bins,
        const iType no_of_flat_bins, histoType *out, fType *max_in, fType *min_in)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;

    // initialize temporary histogram for each block in global memory
    histoType *gmem2 = out + no_of_flat_bins * blockIdx.x;
    // Each thread writes zeros to global memory
    for(unsigned int i = tid; i < no_of_flat_bins; i += blockDim.x)
    {
        gmem2[i] = 0;
    }
    const fType *in[3] = {in_x, in_y, in_z};
    __syncthreads();
    // Process input data by updating the histogram of each block in global
    // memory. Each thread processes one element with all its dimensions at a
    // time.
    for(unsigned int i = gid; i < length; i+=total_threads)
    {
        int current_bin = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType bin_width = (max_in[d]-min_in[d])/no_of_bins[d];
            fType val = in[d][i];
            // Get the bin in the current dimension
            if(val < min_in[d] || val > max_in[d])
            {
                current_bin = no_of_flat_bins+1;
                break;
            }
            int tmp_bin = (val-min_in[d])/bin_width;
            if(tmp_bin >= no_of_bins[d]) tmp_bin--;
            // Get the right place in the histogram
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAdd(&gmem2[current_bin], 1);
        }
    }
}

// Takes edges for each dimension and the number of bins and
// returns a histogram with equally sized bins.
__global__ void histogram_gmem_atomics_with_edges2(const fType *in_x,
        const iType length, const fType *in_y, const fType *in_z,
        const iType no_of_dimensions,
        const iType *no_of_bins, const iType no_of_flat_bins,
        histoType *out, const fType *edges_in)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    unsigned int threads_per_block = blockDim.x;

    // initialize temporary histogram for each block in global memory
    histoType *gmem2 = out + no_of_flat_bins * blockIdx.x;
    // Each thread writes zeros to global memory
    for(unsigned int i = tid; i < no_of_flat_bins; i += threads_per_block)
    {
        gmem2[i] = 0;
    }
    const fType *in[3] = {in_x, in_y, in_z};
    __syncthreads();

    // Process input data by updating the histogram of each block in global
    // memory.
    for(unsigned int i = gid; i < length; i += total_threads)
    {
        int current_bin = 0;
        int bins_offset = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType val = in[d][i];
            int tmp_bin = 0;
            while(val > edges_in[bins_offset+tmp_bin+1]
                && tmp_bin < no_of_bins[d]-1)
            {
                 tmp_bin++;
            }
            if(val > edges_in[bins_offset+tmp_bin+1] || val < edges_in[bins_offset])
            {
                // printf("Thread %%d with value %%f and edges %%f, %%f, length %%d, d %%d, element %%d\n", gid, val,
                //     edges_in[bins_offset+tmp_bin+1], edges_in[bins_offset], length, d, i);
                bins_offset += no_of_bins[d]+1;
                current_bin = no_of_flat_bins + 1;
                break;
            }
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
            bins_offset += no_of_bins[d]+1;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAdd(&gmem2[current_bin], 1);
        }
    }
}

// Takes max and min value for each dimension and the number of bins and
// returns a histogram with equally sized bins.
__global__ void histogram_gmem_atomics_weights2(const fType *in_x,
        const iType length, const fType *in_y, const fType *in_z,
        const iType no_of_dimensions,  const iType *no_of_bins,
        const iType no_of_flat_bins, fType *out, fType *max_in, fType *min_in,
        const fType *weights)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;

    // initialize temporary histogram for each block in global memory
    fType *gmem = out + no_of_flat_bins * blockIdx.x;
    // Each thread writes zeros to global memory
    for(unsigned int i = tid; i < no_of_flat_bins; i += blockDim.x)
    {
        gmem[i] = 0;
    }
    const fType *in[3] = {in_x, in_y, in_z};
    __syncthreads();
    // Process input data by updating the histogram of each block in global
    // memory. Each thread processes one element with all its dimensions at a
    // time.
    for(unsigned int i = gid; i < length; i+=total_threads)
    {
        int current_bin = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType bin_width = (max_in[d]-min_in[d])/no_of_bins[d];
            fType val = in[d][i];
            // Get the bin in the current dimension
            if(val < min_in[d] || val > max_in[d])
            {
                current_bin = no_of_flat_bins+1;
                break;
            }
            int tmp_bin = (val-min_in[d])/bin_width;
            if(tmp_bin >= no_of_bins[d]) tmp_bin--;
            // Get the right place in the histogram
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAddfType(&gmem[current_bin], weights[i]);
        }
    }
}

// Takes edges for each dimension and the number of bins and
// returns a histogram with equally sized bins.
__global__ void histogram_gmem_atomics_with_edges_weights2(const fType *in_x,
        const iType length, const fType *in_y, const fType *in_z,
        const iType no_of_dimensions,
        const iType *no_of_bins, const iType no_of_flat_bins,
        fType *out, const fType *edges_in, const fType *weights)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    unsigned int threads_per_block = blockDim.x;

    // initialize temporary histogram for each block in global memory
    fType *gmem = out + no_of_flat_bins * blockIdx.x;
    // Each thread writes zeros to global memory
    for(unsigned int i = tid; i < no_of_flat_bins; i += threads_per_block)
    {
        gmem[i] = 0;
    }
    const fType *in[3] = {in_x, in_y, in_z};
    __syncthreads();

    // Process input data by updating the histogram of each block in global
    // memory.
    for(unsigned int i = gid; i < length; i += total_threads)
    {
        int current_bin = 0;
        int bins_offset = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType val = in[d][i];
            int tmp_bin = 0;
            while(val > edges_in[bins_offset+tmp_bin+1]
                && tmp_bin < no_of_bins[d]-1)
            {
                 tmp_bin++;
            }
            if(val > edges_in[bins_offset+tmp_bin+1] || val < edges_in[bins_offset])
            {
                bins_offset += no_of_bins[d]+1;
                current_bin = no_of_flat_bins + 1;
                break;
            }
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
            bins_offset += no_of_bins[d]+1;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAddfType(&gmem[current_bin], weights[i]);
        }
    }
}

__global__ void histogram_smem_atomics2(const fType *in_x,
        const iType length, const fType *in_y, const fType *in_z,
        const iType no_of_dimensions,  const iType *no_of_bins,
        const iType no_of_flat_bins, histoType *out, fType *max_in, fType *min_in)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    unsigned int threads_per_block = blockDim.x;

    // initialize temporary accumulation array in shared memory
    extern __shared__ histoType smem2[];
    for(unsigned int i = tid; i < no_of_flat_bins;  i+= threads_per_block)
    {
        smem2[i] = 0;
    }
    const fType *in[3] = {in_x, in_y, in_z};
    __syncthreads();

    // Process input data by updating the histogram of each block in global
    // memory. Each thread processes one element with all its dimensions at a
    // time.
    for(unsigned int i = gid; i < length; i+=total_threads)
    {
        int current_bin = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType bin_width = (max_in[d]-min_in[d])/no_of_bins[d];
            fType val = in[d][i];
            // Get the bin in the current dimension
            if(val < min_in[d] || val > max_in[d])
            {
                current_bin = no_of_flat_bins+1;
                break;
            }
            int tmp_bin = (val-min_in[d])/bin_width;
            if(tmp_bin >= no_of_bins[d]) tmp_bin--;
            // Get the right place in the histogram
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAdd(&smem2[current_bin], 1);
        }
    }
    __syncthreads();
    // Write partial histograms in global memory
    out = &out[blockIdx.x * no_of_flat_bins];
    for(unsigned int i = tid; i < no_of_flat_bins;  i+= threads_per_block)
    {
        out[i] = smem2[i];
    }
}

__global__ void histogram_smem_atomics_with_edges2(const fType *in_x,
        const iType length, const fType *in_y, const fType *in_z,
        const iType no_of_dimensions,
        const iType *no_of_bins, const iType no_of_flat_bins,
        histoType *out, const fType *edges_in)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    unsigned int threads_per_block = blockDim.x;

    // initialize temporary accumulation array in shared memory
    extern __shared__ histoType smem2[];
    for(unsigned int i = tid; i < no_of_flat_bins;  i += threads_per_block)
    {
        smem2[i] = 0;
    }
    const fType *in[3] = {in_x, in_y, in_z};
    __syncthreads();

    // Process input data by updating the histogram of each block in global
    // memory. Each thread processes one element with all its dimensions at a
    // time.
    for(unsigned int i = gid; i < length; i += total_threads)
    {
        int current_bin = 0;
        int bins_offset = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType val = in[d][i];
            int tmp_bin = 0;
            while(val > edges_in[bins_offset+tmp_bin+1]
                    && tmp_bin < no_of_bins[d]-1)
            {
                tmp_bin++;
            }
            if(val > edges_in[bins_offset+tmp_bin+1] || val < edges_in[bins_offset])
            {
                bins_offset += no_of_bins[d]+1;
                current_bin = no_of_flat_bins + 1;
                break;
            }
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
            bins_offset += no_of_bins[d]+1;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAdd(&smem2[current_bin], 1);
        }
    }
    __syncthreads();

    // Write partial histograms in global memory
    histoType *overall_out = &out[blockIdx.x * no_of_flat_bins];
    for(unsigned int i = tid; i < no_of_flat_bins;  i+= threads_per_block)
    {
        overall_out[i] = smem2[i];
    }
}

__global__ void histogram_smem_atomics_weights2(const fType *in_x,
        const iType length, const fType *in_y, const fType *in_z,
        const iType no_of_dimensions,  const iType *no_of_bins,
        const iType no_of_flat_bins, fType *out, fType *max_in, fType *min_in,
        const fType *weights)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    unsigned int threads_per_block = blockDim.x;

    // initialize temporary accumulation array in shared memory
    extern __shared__ fType smem[];
    for(unsigned int i = tid; i < no_of_flat_bins;  i+= threads_per_block)
    {
        smem[i] = 0;
    }
    const fType *in[3] = {in_x, in_y, in_z};
    __syncthreads();

    // Process input data by updating the histogram of each block in global
    // memory. Each thread processes one element with all its dimensions at a
    // time.
    for(unsigned int i = gid; i < length; i+=total_threads)
    {
        int current_bin = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType bin_width = (max_in[d]-min_in[d])/no_of_bins[d];
            fType val = in[d][i];
            // Get the bin in the current dimension
            if(val < min_in[d] || val > max_in[d])
            {
                current_bin = no_of_flat_bins+1;
                break;
            }
            int tmp_bin = (val-min_in[d])/bin_width;
            if(tmp_bin >= no_of_bins[d]) tmp_bin--;
            // Get the right place in the histogram
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAddfType(&smem[current_bin], weights[i]);
        }
    }
    __syncthreads();
    // Write partial histograms in global memory
    out = &out[blockIdx.x * no_of_flat_bins];
    for(unsigned int i = tid; i < no_of_flat_bins;  i+= threads_per_block)
    {
        out[i] = smem[i];
    }
}

__global__ void histogram_smem_atomics_with_edges_weights2(const fType *in_x,
        const iType length, const fType *in_y, const fType *in_z,
        const iType no_of_dimensions,
        const iType *no_of_bins, const iType no_of_flat_bins,
        fType *out, const fType *edges_in, const fType *weights)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int total_threads = blockDim.x * gridDim.x;
    unsigned int threads_per_block = blockDim.x;

    // initialize temporary accumulation array in shared memory
    extern __shared__ fType smem[];
    for(unsigned int i = tid; i < no_of_flat_bins;  i += threads_per_block)
    {
        smem[i] = 0;
    }
    const fType *in[3] = {in_x, in_y, in_z};
    __syncthreads();

    // Process input data by updating the histogram of each block in global
    // memory. Each thread processes one element with all its dimensions at a
    // time.
    for(unsigned int i = gid; i < length; i += total_threads)
    {
        int current_bin = 0;
        int bins_offset = 0;
        for(unsigned int d = 0; d < no_of_dimensions; d++)
        {
            fType val = in[d][i];
            int tmp_bin = 0;
            while(val > edges_in[bins_offset+tmp_bin+1]
                    && tmp_bin < no_of_bins[d]-1)
            {
                tmp_bin++;
            }
            if(val > edges_in[bins_offset+tmp_bin+1] || val < edges_in[bins_offset])
            {
                bins_offset += no_of_bins[d]+1;
                current_bin = no_of_flat_bins + 1;
                break;
            }
            int power_bins = 1;
            for(unsigned int k=no_of_dimensions-1; k > d; k--)
            {
                power_bins = no_of_bins[k] * power_bins;
            }
            current_bin += tmp_bin * power_bins;
            bins_offset += no_of_bins[d]+1;
        }
        // Avoid illegal memory access
        if(current_bin < no_of_flat_bins)
        {
            atomicAddfType(&smem[current_bin], weights[i]);
        }
    }
    __syncthreads();

    // Write partial histograms in global memory
    fType *overall_out = &out[blockIdx.x * no_of_flat_bins];
    for(unsigned int i = tid; i < no_of_flat_bins;  i+= threads_per_block)
    {
        overall_out[i] = smem[i];
    }
}
