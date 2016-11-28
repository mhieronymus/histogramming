/*
 * author: M. Hieronymus
 *
 * date: November 2016
 *
 */
#ifndef __HISTOGRAM_ATOMICSHINCLUDED__
#define __HISTOGRAM_ATOMICSHINCLUDED__



__device__ fType __ull_as_fType(unsigned long long a)
{
    union {unsigned long long a; fType b;} u;
    u.a = a;
    return u.b;
}

__device__ fType __fType_as_ull(fType a)
{
    union {fType a; unsigned long long b;} u;
    u.a = a;
    return u.b;
}
__device__ fType atomicMaxAny(fType *adress, fType val);
__device__ fType atomicMinAny(fType *adress, fType val);
__global__ void max_min_reduce(const fType *d_array, const size_t length,
        fType d_max, fType d_min);

__global__ void histogram_gmem_atomics(const fType *in,  const size_t length,
        const size_t no_of_dimensions,  const size_t no_of_bins,
        unsigned int *out);


// __global__ void histogram_smem_atomics(const fType *in, int width,
//         int height, unsigned int *out);
__global__ void histogram_final_accum(const unsigned int *in,
        int no_of_histograms, unsigned int *out,
        unsigned int no_of_bins, unsigned int no_of_dimensions);

#endif
