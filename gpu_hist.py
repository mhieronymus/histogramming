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
    FTYPE : np.float64 or np.float32
    """

    FTYPE = np.float64
    C_FTYPE = 'double'
    C_PRECISION_DEF = 'DOUBLE_PRECISION'
    C_ITYPE = 'unsigned int'
    C_HIST_TYPE = 'unsigned int'
    HIST_TYPE = np.uint32
    ITYPE = np.uint32

    def __init__(self, FTYPE=np.float64):
        # Set some default types.
        self.FTYPE = FTYPE
        # Might be useful. PISA used it for atomic cuda_utils.h with
        # custom atomic_add for floats and doubles.
        #include_dirs = [os.path.abspath(find_resource('../gpu_hist'))]
        kernel_code = open("gpu_hist/histogram_atomics.cu", "r").read() %dict(
            c_precision_def=self.C_PRECISION_DEF,
            c_ftype=self.C_FTYPE,
            c_itype=self.C_ITYPE,
            c_uitype=self.C_HIST_TYPE
        )
        include_dirs = ['/gpu_hist']
        # keep for compiler output, no_extern_c: allow name manling
        module = SourceModule(kernel_code, keep=True,
                options=['--compiler-options','-Wall', '-g'],
                include_dirs=include_dirs, no_extern_c=False)
        #module = SourceModule(kernel_code, include_dirs=include_dirs, keep=True)
        self.max_min_reduce = module.get_function("max_min_reduce")
        self.hist_gmem = module.get_function("histogram_gmem_atomics")
        self.hist_gmem_given_edges = module.get_function("histogram_gmem_atomics_with_edges")
        self.hist_smem = module.get_function("histogram_smem_atomics")
        self.hist_smem_given_edges = module.get_function("histogram_smem_atomics_with_edges")
        self.hist_accum = module.get_function("histogram_final_accum")

        gpu_attributes = cuda.Device(0).get_attributes()
        # See https://documen.tician.de/pycuda/driver.html
        self.max_threads_per_block = gpu_attributes.get(
                cuda.device_attribute.MAX_THREADS_PER_BLOCK)
        self.max_block_dim_x = gpu_attributes.get(
                cuda.device_attribute.MAX_BLOCK_DIM_X)
        self.max_grid_dim_x = gpu_attributes.get(
                cuda.device_attribute.MAX_GRID_DIM_X)
        self.warp_size = gpu_attributes.get(
                cuda.device_attribute.WARP_SIZE)
        self.shared_memory = gpu_attributes.get(
                cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
        self.constant_memory = gpu_attributes.get(
                cuda.device_attribute.TOTAL_CONSTANT_MEMORY)
        self.threads_per_mp = gpu_attributes.get(
                cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR)
        self.mp = gpu_attributes.get(
                cuda.device_attribute.MULTIPROCESSOR_COUNT)
        self.memory, total = cuda.mem_get_info()

        print "################################################################"
        print "Your device has following attributes:"
        print "Max threads per block: ", self.max_threads_per_block
        print "Max x-dimension for block : ", self.max_block_dim_x
        print "Max x-dimension for grid: ", self.max_grid_dim_x
        print "Warp size: ", self.warp_size
        print "Max shared memory per block: ", self.shared_memory/1024, "Kbytes"
        print "Total constant memory: ", self.constant_memory/1024, "Kbytes"
        print "Max threads per multiprocessor: ", self.threads_per_mp
        print "Number of multiprocessors: ", self.mp
        print "Available global memory: ", self.memory/(1024*1024), " Mbytes"
        print "################################################################"


    def clear(self):
        """Clear the histogram bins on the GPU"""
        self.hist = np.zeros(self.n_flat_bins, dtype=self.HIST_TYPE)
        cuda.memcpy_htod(self.d_hist, self.hist)


    # TODO: Check if shared memory or not
    def get_hist(self, n_events, shared=True, bins=10, normed=False, weights=None):
        """Retrive histogram with given events and edges

        Parameters
        ----------
        bins: If edges, than with the rightmost edge!
        """
        no_of_dimensions = self.ITYPE(len(n_events[0]))
        edges = None
        bins_per_dimension = None
        n_edges = None
        if type(bins) is int:
            no_of_bins = self.ITYPE(bins)
            self.n_flat_bins = self.ITYPE(no_of_bins ** no_of_dimensions)
        elif type(bins[0]) is not list and type(bins[0]) is not np.ndarray:
            self.n_flat_bins = 1
            for i in range(len(bins)):
                self.n_flat_bins = self.n_flat_bins * bins[0]
            bins_per_dimension = bins
        else:
            self.n_flat_bins= 1
            for i in range(len(bins)):
                self.n_flat_bins = self.n_flat_bins * (len(bins[i]) - 1)
            self.n_flat_bins = self.ITYPE(self.n_flat_bins)
            no_of_bins = self.ITYPE(len(bins[0])-1)
            edges = bins
            n_edges = sum(sum(1 for i in b if i) for b in bins)

        # We use a one-dimensional block and grid.
        # We use as many threads per block as possible but we are limited
        # to the shared memory.
        no_of_threads = (self.shared_memory /
                    np.dtype(self.C_FTYPE).itemsize * 2)
        if no_of_threads > self.max_threads_per_block:
            overflow = self.max_threads_per_block%no_of_dimensions
            self.block_dim = (self.max_threads_per_block-overflow, 1, 1)
        else:
            overflow = no_of_threads%no_of_dimensions
            self.block_dim = (no_of_threads-overflow, 1, 1)
        # debug
        # overflow = 4%no_of_dimensions
        # self.block_dim = (4-overflow, 1, 1)

        self.hist = np.zeros(self.n_flat_bins, dtype=self.HIST_TYPE)
        self.d_hist = cuda.mem_alloc(self.n_flat_bins
                * np.dtype(self.HIST_TYPE).itemsize)
        # Define shared memory for max- and min-reduction
        self.shared = (self.block_dim[0] * np.dtype(self.C_FTYPE).itemsize * 2)

        # Check if shared memory can be used
        if shared:
            if (self.n_flat_bins * np.dtype(self.HIST_TYPE).itemsize) > self.shared_memory:
                shared = False
                print "Not enough shared memory available. Switching to global memory usage."
        # Copy the  arrays
        d_events = cuda.mem_alloc(n_events.nbytes)
        cuda.memcpy_htod(d_events, n_events)

        # Calculate the number of blocks needed
        dx, mx = divmod(len(n_events), self.block_dim[0])
        self.grid_dim = ( (dx + (mx>0)), 1 )

        # Allocate local histograms on device
        d_tmp_hist = cuda.mem_alloc(self.n_flat_bins * self.grid_dim[0]
                * np.dtype(self.HIST_TYPE).itemsize)
        if shared:
            # Calculate edges by yourself if no edges are given
            if edges is None and bins_per_dimension is None:
                d_max_in = cuda.mem_alloc(no_of_dimensions
                        * np.dtype(self.FTYPE).itemsize)
                d_min_in = cuda.mem_alloc(no_of_dimensions
                        * np.dtype(self.FTYPE).itemsize)
                self.max_min_reduce(d_events,
                        self.HIST_TYPE(len(n_events)),
                        no_of_dimensions, d_max_in, d_min_in,
                        block=self.block_dim, grid=self.grid_dim,
                        shared=self.shared)
                # Calculate local histograms on shared memory on device
                self.shared = (self.n_flat_bins * np.dtype(self.HIST_TYPE).itemsize)
                self.hist_smem(d_events,
                        self.HIST_TYPE(len(n_events)*no_of_dimensions),
                        no_of_dimensions, no_of_bins, self.n_flat_bins,
                        d_tmp_hist, d_max_in, d_min_in,
                        block=self.block_dim, grid=self.grid_dim,
                        shared=self.shared)
            elif bins_per_dimension is None:
                self.shared = (self.n_flat_bins * np.dtype(self.HIST_TYPE).itemsize)
                d_edges_in = cuda.mem_alloc(n_edges
                        * np.dtype(self.FTYPE).itemsize)
                cuda.memcpy_htod(d_edges_in, edges)
                self.hist_smem_given_edges(d_events,
                        self.HIST_TYPE(len(n_events)*no_of_dimensions),
                        no_of_dimensions, no_of_bins, self.n_flat_bins,
                        d_tmp_hist, d_edges_in,
                        block=self.block_dim, grid=self.grid_dim,
                        shared=self.shared)
                # # Debug
                # tmp_hist = np.zeros(self.n_flat_bins * self.grid_dim[0], dtype=self.HIST_TYPE)
                # cuda.memcpy_dtoh(tmp_hist, d_tmp_hist)
                # tmp_hist = np.reshape(tmp_hist, (self.grid_dim[0], no_of_bins, no_of_bins))
                # print np.sum(tmp_hist)
                # print "tmp_hist:\n", tmp_hist
            else:
                print "Different amount of bins per dimension is not implemented"
        else:
            # Calculate edges by yourself if no edges are given
            if edges is None and bins_per_dimension is None:
                d_max_in = cuda.mem_alloc(no_of_dimensions
                        * np.dtype(self.FTYPE).itemsize)
                d_min_in = cuda.mem_alloc(no_of_dimensions
                        * np.dtype(self.FTYPE).itemsize)
                self.max_min_reduce(d_events,
                        self.HIST_TYPE(len(n_events)),
                        no_of_dimensions, d_max_in, d_min_in,
                        block=self.block_dim, grid=self.grid_dim,
                        shared=self.shared)
                self.hist_gmem(d_events,
                        self.HIST_TYPE(len(n_events)*no_of_dimensions),
                        no_of_dimensions, no_of_bins, self.n_flat_bins,
                        d_tmp_hist, d_max_in, d_min_in,
                        block=self.block_dim, grid=self.grid_dim)
                # Debug
                # tmp_hist = np.zeros(self.n_flat_bins * self.grid_dim[0], dtype=self.HIST_TYPE)
                # cuda.memcpy_dtoh(tmp_hist, self.d_tmp_hist)
                # tmp_hist = np.reshape(tmp_hist, (self.grid_dim[0], no_of_bins, no_of_bins))
                # print np.sum(tmp_hist)
                # print "tmp_hist:\n", tmp_hist
            elif bins_per_dimension is None:
                d_edges_in = cuda.mem_alloc(n_edges
                        * np.dtype(self.FTYPE).itemsize)
                cuda.memcpy_htod(d_edges_in, edges)
                self.hist_gmem_given_edges(d_events,
                        self.HIST_TYPE(len(n_events)*no_of_dimensions),
                        no_of_dimensions, no_of_bins, self.n_flat_bins,
                        d_tmp_hist, d_edges_in,
                        block=self.block_dim, grid=self.grid_dim)
            else:
                print "Different amount of bins per dimension is not implemented"
        self.hist_accum(d_tmp_hist, self.ITYPE(self.grid_dim[0]), self.d_hist,
                no_of_bins, self.n_flat_bins, no_of_dimensions,
                block=self.block_dim, grid=self.grid_dim)
        # Copy the array back and make the right shape
        cuda.memcpy_dtoh(self.hist, self.d_hist)
        histo_shape = ()
        for d in range(0, no_of_dimensions):
            histo_shape += (no_of_bins, )
        self.hist = np.reshape(self.hist, histo_shape)

        if edges is None and bins_per_dimension is None:
            # Calculate the found edges
            max_in = np.zeros(no_of_dimensions, dtype=self.FTYPE)
            min_in = np.zeros(no_of_dimensions, dtype=self.FTYPE)
            cuda.memcpy_dtoh(max_in, d_max_in)
            cuda.memcpy_dtoh(min_in, d_min_in)
            edges = []
            # Create some nice edges
            for d in range(0, no_of_dimensions):
                bin_width = (max_in[d]-min_in[d])/no_of_bins
                edges_d =  np.arange(min_in[d], max_in[d]+1, bin_width, dtype=self.FTYPE)
                edges.append(edges_d)
        elif edges is None:
            print "Different amount of bins per dimension is not implemented"
        return self.hist, edges


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


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        #self.clear()
        return


def test_GPUHist():
    """A small test which calculates a histogram"""

if __name__ == '__main__':
    test_GPUHist()
