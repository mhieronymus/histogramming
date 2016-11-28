# authors: M. Hieronymus (mhierony@students.uni-mainz.de)
# date:    November 2016

import os

import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.autoinit

# from pisa import FTYPE, C_FTYPE, C_PRECISION_DEF # Used in PISA

#TODO: Add more comments. Remove edges array and add multidimensional edge array.
class GPUHist(object):
    """
    Histogramming class for GPUs
    Basic implemention is based on
    https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
    and modified by M. Hieronymus.

    Parameters
    ----------
    bin_edges_x : array
    bin_edges_y : array
    bin_edges_z : array (optional)
    FTYPE : np.float64 or np.float32
    """

    FTYPE = np.float64
    C_FTYPE = 'double'
    C_PRECISION_DEF = 'DOUBLE_PRECISION'
    C_ITYPE = 'long'
    ITYPE = np.int64

    #TODO: Remove three bin edges and add multidimensional bin edge.
    def __init__(self, no_of_dimensions, no_of_bins, FTYPE=np.float64):
        #TODO: Check for dimensions, memory, GPU version, FType
        # Set some default types.
        self.FTYPE = FTYPE
        # Might be useful. PISA used it for atomic cuda_utils.h with
        # custom atomic_add for floats and doubles.
        #include_dirs = [os.path.abspath(find_resource('../gpu_hist'))]
        #TODO: Add self.n_flat_bins = product of length of all dimensions
        self.n_flat_bins = no_of_dimensions * no_of_bins
        self.no_of_dimensions = no_of_dimensions
        self.no_of_bins = no_of_bins
        kernel_code = open("gpu_hist/histogram_atomics.cuh", "r").read() %dict(
            c_precision_def=self.C_PRECISION_DEF,
            c_ftype=self.C_FTYPE,
            c_itype=self.C_ITYPE,
            n_flat_bins=self.n_flat_bins
        )
        include_dirs = ['gpu_hist/']
        # keep for compiler output, no_extern_c: allow name manling
        module = SourceModule(kernel_code, keep=True,
                options=['--compiler-options','-Wall'], include_dirs=include_dirs, no_extern_c=False)
        #module = SourceModule(kernel_code, include_dirs=include_dirs, keep=True)
        self.hist_gmem = module.get_function("histogram_gmem_atomics")
        self.hist_accum = module.get_function("histogram_final_accum")


    def clear(self, grid_dim):
        """Clear the histogram bins on the GPU"""
        self.hist = np.zeros(self.n_flat_bins, dtype=FTYPE)
        cuda.memcpy_htod(self.d_hist, self.hist)
        tmp_hist = np.zeros(self.n_flat_bins * grid_dim, dtype=FTYPE)
        cuda.memcpy_htod(self.d_tmp_hist, tmp_hist)

    # TODO: Check if shared memory or not
    def get_hist(self, n_events, n_flat_bins, shared):
        """Retrive histogram with given events and edges"""
        # TODO: Calculate block and grid dimensions
        block_dim = 16
        grid_dim = 4
        self.clear(grid_dim)
        # Copy the  arrays
        d_events = cuda.mem_alloc(n_events.nbytes)
        cuda_memcpy_htod(d_events, n_events)

        self.hist_gmem(n_events, len(n_events), no_of_dimensions, no_of_bins,
                d_tmp_hist, block=block_dim, grid=grid_dim)
        # TODO: Check if new dimensions might be useful
        self.hist_accum(d_tmp_hist, grid_dim, self.d_hist, self.no_of_bins,
                self.no_of_dimensions, block=block_dim, grid=grid_dim)
        # Copy the array back
        cuda.memcpy_dtoh(self.hist, self.d_hist)
        # TODO: reshape according to given dimensions.
        #TODO: Check if device arrays are given
        # Calculate histogram
        # Copy the results back to host
        return hist


    def set_variables(FTYPE):
        """This method sets some variables like FTYPE and should be called at
        least once before calculating a histogram. Those variables are already
        set in PISA with the commented import from above."""
        if FTYPE == np.float32:
            C_FTYPE = 'float'
            C_PRECISION_DEF = 'SINGLE_PRECISION'
            self.FTYPE = FTYPE
            sys.stderr.write("Histogramming is set to single precision (FP32) "
                    "mode.\n\n")
        elif FTYPE == np.float64:
            C_FTYPE = 'double'
            C_PRECISION_DEF = 'DOUBLE_PRECISION'
            self.FTYPE = FTYPE
            sys.stderr.write("Histogramming is set to double precision (FP64) "
                    "mode.\n\n")
        else:
            raise ValueError('FTYPE must be one of `np.float32` or `np.float64`'
                    '. Got %s instead.' %FTYPE)

def test_GPUHist():
    """A small test which calculates a histogram"""

if __name__ == '__main__':
    test_GPUHist()
