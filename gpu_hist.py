# authors: M. Hieronymus (mhierony@students.uni-mainz.de)
# date:    November 2016

import os

import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda

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
    def __init__(self, bin_edges, FTYPE=np.float64):
        #TODO: Check for dimensions, memory, GPU version, FType
        # Set some default types.
        set_variables(FTYPE)
        # Might be useful. PISA used it for atomic cuda_utils.h with
        # custom atomic_add for floats and doubles.
        include_dirs = [os.path.abspath(find_resource('../gpu_hist'))]
        #TODO: Add self.n_flat_bins = product of length of all dimensions
        kernel_code = open("gpu_hist/histogram_atomics.cuh", "r").read() %dict(
            c_precision_def=self.C_PRECISION_DEF,
            c_ftype=self.C_FTYPE,
            c_itype=self.C_ITYPE,
            n_flat_bins=self.n_flat_bins,
            events_per_thread=self.events_per_thread
        )
        module = SourceModule(kernel_code, include_dirs=include_dirs, keep=True)

    def clear(self):
        """Clear the histogram bins on the GPU"""

    def get_hist(self, n_events, d_x, d_y, d_w, d_z=None):
        """Retrive histogram with given device arrays x and y and weights w"""

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
