/*
 * author: M. Hieronymus
 *
 * date: November 2016
 *
 */
#ifndef __HISTOGRAM_ATOMICSHINCLUDED__
#define __HISTOGRAM_ATOMICSHINCLUDED__

#define %(c_precision_def)s
#define fType %(c_ftype)s
#define iType %(c_itype)s
#define N_FLAT_BINS %(n_flat_bins)i
#define EVENTS_PER_THREAD %(events_per_thread)i

__global__ void histogram_gmem_atomics(const IN_TYPE *in, int width,
        int height, unsigned int *out);
__global__ void histogram_smem_atomics(const IN_TYPE *in, int width,
        int height, unsigned int *out);
__global__ void histogram_final_accum(const unsigned int *in, int n,
        unsigned int *out);

#endif
