"""Creating histograms with GPUs.

This module calculates histograms using the GPU. A more detailed description
can be found here: https://github.com/PolygonAndPixel/histogramming/wiki

The interface is very similar to
[`numpy.histogramdd`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogramdd.html).
If you are using `histogramdd` already you shouldn't have a lot of work to
include `gpu_hist.py`.
You need to instantiate `gpu_hist` once by using:

    `histogrammer = gpu_hist.GPUHist()`

`gpu_hist` uses `np.float64` by default but you can change the precision of your events by using

    `histogrammer = gpu_hist.GPUHist(np.float32)`

After that you can get your histogram with:

    gpu_hist.get_hist(sample, shared=True, bins=10, normed=False,
                      weights=None, dimensions=1, number_of_events=0)

authors: M. Hieronymus (mhierony@students.uni-mainz.de)
date:    February 2017
"""

import sys
import time

import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.autoinit

FTYPE = np.float64

class GPUHist(object):
    """
    Histogramming class for GPUs
    Basic implemention is based on
    https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
    and modified by M. Hieronymus.
    Parameters
    ----------
    ftype : np.float64 or np.float32
    """

    def __init__(self, ftype=FTYPE):
        t0 = time.time()

        self.FTYPE = ftype
        self.C_ITYPE = 'unsigned int'
        self.ITYPE = np.uint32
        self.HIST_TYPE = np.uint32
        self.C_HIST_TYPE = 'unsigned int'

        # Set some default types.
        if ftype == np.float32:
            self.C_FTYPE = 'float'
            self.C_PRECISION_DEF = 'SINGLE_PRECISION'
            self.C_CHANGETYPE = 'int'
        elif ftype == np.float64:
            self.C_FTYPE = 'double'
            self.C_PRECISION_DEF = 'DOUBLE_PRECISION'
            self.C_CHANGETYPE = 'unsigned long long int'
        else:
            raise ValueError('Invalid `ftype` specified; must be either'
                             ' `numpy.float32` or `numpy.float64`')
        # Might be useful. PISA used it for atomic cuda_utils.h with

        # custom atomic_add for floats and doubles.
        #include_dirs = [os.path.abspath(find_resource('../gpu_hist'))]
        kernel_code = open("gpu_hist/histogram_atomics.cu", "r").read() %dict(
            c_precision_def=self.C_PRECISION_DEF,
            c_ftype=self.C_FTYPE,
            c_itype=self.C_ITYPE,
            c_uitype=self.C_HIST_TYPE,
            c_changetype=self.C_CHANGETYPE
        )
        include_dirs = ['/gpu_hist']
        # keep for compiler output, no_extern_c: allow name manling
        # Add -g for debug mode
        module = SourceModule(kernel_code, keep=True,
                              options=['--compiler-options', '-Wall'],
                              include_dirs=include_dirs, no_extern_c=False)
        self.max_min_reduce = module.get_function("max_min_reduce")
        self.hist_gmem = module.get_function("histogram_gmem_atomics")
        self.hist_gmem_given_edges = module.get_function("histogram_gmem_atomics_with_edges")
        self.hist_gmem_weights = module.get_function("histogram_gmem_atomics_weights")
        self.hist_gmem_given_edges_weights = module.get_function("histogram_gmem_atomics_with_edges_weights")

        self.hist_smem = module.get_function("histogram_smem_atomics")
        self.hist_smem_given_edges = module.get_function("histogram_smem_atomics_with_edges")
        self.hist_smem_weights = module.get_function("histogram_smem_atomics_weights")
        self.hist_smem_given_edges_weights = module.get_function("histogram_smem_atomics_with_edges_weights")

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
        self.memory = cuda.mem_get_info()[0]

        # print "################################################################"
        # print "Your device has following attributes:"
        # print "Max threads per block: ", self.max_threads_per_block
        # print "Max x-dimension for block : ", self.max_block_dim_x
        # print "Max x-dimension for grid: ", self.max_grid_dim_x
        # print "Warp size: ", self.warp_size
        # print "Max shared memory per block: ", self.shared_memory/1024, "Kbytes"
        # print "Total constant memory: ", self.constant_memory/1024, "Kbytes"
        # print "Max threads per multiprocessor: ", self.threads_per_mp
        # print "Number of multiprocessors: ", self.mp
        # print "Available global memory: ", self.memory/(1024*1024), " Mbytes"
        # print "################################################################"
        self.init_time = time.time() - t0


    def clear(self):
        """Clear the histogram bins on the GPU"""
        self.hist = np.zeros(self.n_flat_bins, dtype=self.HIST_TYPE)
        cuda.memcpy_htod(self.d_hist, self.hist)


    def get_hist(self, sample, shared=True, bins=10, normed=False,
                 weights=None, dims=1, number_of_events=0):
        """Retrive histogram with given events and edges

        Parameters
        ----------
        bins: If edges, than with the rightmost edge!
        dimensions: If a device array is given, provide the number of dims

        Returns
        -------

        """
        t0 = time.time()

        if isinstance(sample, cuda.DeviceAllocation):
            if number_of_events > 0:
                n_dims = dims
                n_events = number_of_events
            else:
                raise ValueError("If you use a device array as input, you have "
                                 "to specify the number of events in your input "
                                 "and the number of dimensions (default is 1 "
                                 "for dimensions).\n\n")
        else:
            try:
                n_events, n_dims = sample.shape
            except (AttributeError, ValueError):
                sample = np.atleast_2d(sample).T
                n_events, n_dims = sample.shape
            n_dims = self.ITYPE(n_dims)

        edges = None
        d_edges_in = None
        d_max_in = None
        d_min_in = None
        flattened = False

        sizeof_hist_t = np.dtype(self.HIST_TYPE).itemsize
        sizeof_c_ftype = np.dtype(self.C_FTYPE).itemsize
        sizeof_float_t = np.dtype(self.FTYPE).itemsize

        # Check if number of bins for all dimensions is given or
        # if number of bins for each dimension is given or
        # if the edges for each dimension are given
        if isinstance(bins, int):
            # Use equally spaced bins in all dimensions
            self.n_flat_bins = self.ITYPE(self.ITYPE(bins) ** n_dims)
            no_of_bins = [self.ITYPE(bins) for _ in xrange(n_dims)]
            no_of_bins = np.asarray(no_of_bins)
            d_no_of_bins = cuda.mem_alloc(no_of_bins.nbytes)
            cuda.memcpy_htod(d_no_of_bins, no_of_bins)
        elif not isinstance(bins[0], list) and not isinstance(bins[0], np.ndarray):
            # Use different amounts of bins in each dimension
            self.n_flat_bins = 1
            no_of_bins = []
            for b in bins:
                self.n_flat_bins = self.n_flat_bins * b
                no_of_bins.append(self.ITYPE(b))
            no_of_bins = np.asarray(no_of_bins)
            d_no_of_bins = cuda.mem_alloc(no_of_bins.nbytes)
            cuda.memcpy_htod(d_no_of_bins, no_of_bins)
        else:
            # Use given edges
            self.n_flat_bins = 1
            no_of_bins = []
            for b in bins:
                self.n_flat_bins = self.n_flat_bins * (len(b) - 1)
                no_of_bins.append(self.ITYPE(len(b)-1))
            no_of_bins = np.asarray(no_of_bins)
            d_no_of_bins = cuda.mem_alloc(no_of_bins.nbytes)
            cuda.memcpy_htod(d_no_of_bins, no_of_bins)
            self.n_flat_bins = self.ITYPE(self.n_flat_bins)
            if isinstance(bins, list):
                # Different amount of bins for each dimension. Therefore flatten
                # the list before casting to array
                edges = np.asarray([item for sublist in bins for item in sublist])
                flattened = True
            else:
                edges = bins

        # We use a one-dimensional block and grid.
        # We use as many threads per block as possible but we are limited
        # to the shared memory.
        no_of_threads = (self.shared_memory / sizeof_c_ftype * 2)
        if no_of_threads > self.max_threads_per_block:
            overflow = self.max_threads_per_block%n_dims
            self.block_dim = (self.max_threads_per_block-overflow, 1, 1)
        else:
            overflow = no_of_threads%n_dims
            self.block_dim = (no_of_threads-overflow, 1, 1)

        self.hist = np.zeros(self.n_flat_bins, dtype=self.HIST_TYPE)
        self.d_hist = cuda.mem_alloc(self.n_flat_bins * sizeof_hist_t)

        # Define shared memory for max- and min-reduction
        self.shared = (self.block_dim[0] * sizeof_c_ftype * 2)

        # Check if shared memory can be used
        if shared and self.n_flat_bins * sizeof_hist_t > self.shared_memory:
            shared = False
            sys.stderr.write("Not enough shared memory available; "
                             "switching to global memory. "
                             "(n_flat_bins=%d, sizeof_hist_t=%d bytes)\n"
                             % (self.n_flat_bins, sizeof_hist_t))

        # Copy the  arrays
        if isinstance(sample, cuda.DeviceAllocation):
            d_sample = sample
        else:
            d_sample = cuda.mem_alloc(sample.nbytes)
            cuda.memcpy_htod(d_sample, sample)
        if isinstance(weights, cuda.DeviceAllocation):
            d_weights = weights
        else:
            d_weights = cuda.mem_alloc(weights.nbytes)
            cuda.memcpy_htod(d_weights, weights)

        # Calculate the number of blocks needed
        dx, mx = divmod(n_events, self.block_dim[0])
        self.grid_dim = ((dx + (mx > 0)), 1)

        # Allocate local histograms on device
        try:
            d_tmp_hist = cuda.mem_alloc(
                self.n_flat_bins
                * self.grid_dim[0]
                * sizeof_hist_t
            )
        except pycuda._driver.MemoryError:
            available_memory = cuda.mem_get_info()[0]
            print ("Trying to allocate %d Mbytes for temporary histograms. "
                   "Only %d Mbytes available. self.n_flat_bins: %d"
                   " self.grid_dim[0]: %d sizeof_hist_t: %d\n"
                   % (self.n_flat_bins * self.grid_dim[0] * sizeof_hist_t/(1024*1024),
                      available_memory/(1024*1024), self.n_flat_bins,
                      self.grid_dim[0], sizeof_hist_t))
            raise

        if shared:
            # Calculate edges by yourself if no edges are given
            if edges is None:
                d_max_in = cuda.mem_alloc(n_dims * sizeof_float_t)
                d_min_in = cuda.mem_alloc(n_dims * sizeof_float_t)
                self.max_min_reduce(d_sample,
                                    self.HIST_TYPE(n_events),
                                    self.HIST_TYPE(n_dims), d_max_in, d_min_in,
                                    block=self.block_dim, grid=self.grid_dim,
                                    shared=self.shared)
                if weights is None:
                    # Calculate local histograms on shared memory on device
                    self.shared = (self.n_flat_bins * sizeof_hist_t)
                    self.hist_smem(d_sample,
                                   self.HIST_TYPE(n_events*n_dims),
                                   self.HIST_TYPE(n_dims),
                                   d_no_of_bins,
                                   self.HIST_TYPE(self.n_flat_bins),
                                   d_tmp_hist, d_max_in, d_min_in,
                                   block=self.block_dim, grid=self.grid_dim,
                                   shared=self.shared)
                else:
                    # Calculate local histograms with weights
                    self.hist_smem_weights(d_sample,
                                           self.HIST_TYPE(n_events*n_dims),
                                           self.HIST_TYPE(n_dims),
                                           d_no_of_bins,
                                           self.HIST_TYPE(self.n_flat_bins),
                                           d_tmp_hist, d_max_in, d_min_in,
                                           d_weights,
                                           block=self.block_dim,
                                           grid=self.grid_dim,
                                           shared=self.shared)
            else:
                if weights is None:
                    self.shared = (self.n_flat_bins * sizeof_hist_t)
                    d_edges_in = cuda.mem_alloc(edges.nbytes)
                    cuda.memcpy_htod(d_edges_in, edges)
                    self.hist_smem_given_edges(d_sample,
                                               self.HIST_TYPE(n_events*n_dims),
                                               self.HIST_TYPE(n_dims),
                                               d_no_of_bins,
                                               self.HIST_TYPE(self.n_flat_bins),
                                               d_tmp_hist, d_edges_in,
                                               block=self.block_dim,
                                               grid=self.grid_dim,
                                               shared=self.shared)
               else:
                   # Calculate local histograms with edges and weights
                   self.hist_smem_given_edges_weights(d_sample,
                                                      self.HIST_TYPE(n_events*n_dims),
                                                      self.HIST_TYPE(n_dims),
                                                      d_no_of_bins,
                                                      self.HIST_TYPE(self.n_flat_bins),
                                                      d_tmp_hist, d_edges_in,
                                                      d_weights,
                                                      block=self.block_dim,
                                                      grid=self.grid_dim,
                                                      shared=self.shared)
        else:
            # Calculate edges by yourself if no edges are given
            if edges is None:
                d_max_in = cuda.mem_alloc(n_dims * sizeof_float_t)
                d_min_in = cuda.mem_alloc(n_dims * sizeof_float_t)
                self.max_min_reduce(d_sample,
                                    self.HIST_TYPE(n_events),
                                    self.HIST_TYPE(n_dims), d_max_in, d_min_in,
                                    block=self.block_dim, grid=self.grid_dim,
                                    shared=self.shared)
                if weights is None:
                    self.hist_gmem(d_sample,
                                   self.HIST_TYPE(n_events*n_dims),
                                   self.HIST_TYPE(n_dims),
                                   d_no_of_bins,
                                   self.HIST_TYPE(self.n_flat_bins),
                                   d_tmp_hist, d_max_in, d_min_in,
                                   block=self.block_dim, grid=self.grid_dim)
                else:
                    # Calculate global histograms with weights
                    self.hist_gmem_weights(d_sample,
                                           self.HIST_TYPE(n_events*n_dims),
                                           self.HIST_TYPE(n_dims),
                                           d_no_of_bins,
                                           self.HIST_TYPE(self.n_flat_bins),
                                           d_tmp_hist, d_max_in, d_min_in,
                                           d_weights,
                                           block=self.block_dim,
                                           grid=self.grid_dim)
            else:
                d_edges_in = cuda.mem_alloc(edges.nbytes)
                cuda.memcpy_htod(d_edges_in, edges)
                if weights is None:
                    self.hist_gmem_given_edges(d_sample,
                                               self.HIST_TYPE(n_events*n_dims),
                                               self.HIST_TYPE(n_dims),
                                               d_no_of_bins,
                                               self.HIST_TYPE(self.n_flat_bins),
                                               d_tmp_hist, d_edges_in,
                                               block=self.block_dim,
                                               grid=self.grid_dim)
                else:
                    # Calculate global histograms with edges and weights
                    self.hist_gmem_given_edges_weights(d_sample,
                                                       self.HIST_TYPE(n_events*n_dims),
                                                       self.HIST_TYPE(n_dims),
                                                       d_no_of_bins,
                                                       self.HIST_TYPE(self.n_flat_bins),
                                                       d_tmp_hist, d_edges_in,
                                                       d_weights,
                                                       block=self.block_dim,
                                                       grid=self.grid_dim)

        self.hist_accum(d_tmp_hist, self.ITYPE(self.grid_dim[0]), self.d_hist,
                        self.HIST_TYPE(self.n_flat_bins),
                        block=self.block_dim, grid=self.grid_dim)
        # Copy the array back and make the right shape
        cuda.memcpy_dtoh(self.hist, self.d_hist)
        histo_shape = ()
        for d in range(0, n_dims):
            histo_shape += (no_of_bins[d], )
        self.hist = np.reshape(self.hist, histo_shape)

        if edges is None and bins_per_dimension is None:
            # Calculate the found edges
            max_in = np.zeros(n_dims, dtype=self.FTYPE)
            min_in = np.zeros(n_dims, dtype=self.FTYPE)
            cuda.memcpy_dtoh(max_in, d_max_in)
            cuda.memcpy_dtoh(min_in, d_min_in)
            edges = []
            # Create some nice edges
            for d in range(0, n_dims):
                try:
                    edges_d = np.linspace(min_in[d], max_in[d],
                                          no_of_bins[d]+1, dtype=self.FTYPE)
                except ValueError:
                    print min_in[d], max_in[d], no_of_bins[d], self.FTYPE
                    raise
                edges.append(edges_d)

        self.d_hist.free()
        d_tmp_hist.free()
        if not isinstance(sample, cuda.DeviceAllocation):
            # TODO: Move freeing of d_samples to another method (do we actually
            # want to free it?) Also: free samples: Freeing is only possible
            # where it has been created.
            d_sample.free()

        if d_edges_in is not None:
            d_edges_in.free()
        if d_max_in is not None:
            d_max_in.free()
        if d_min_in is not None:
            d_min_in.free()
        # Check if edges had to be flattened before:
        if flattened:
            edges = bins

        self.calc_time = time.time() - t0

        return self.hist, edges


    def set_variables(self, ftype):
        """This method sets some variables like FTYPE and should be called at
        least once before calculating a histogram. Those variables are already
        set in PISA with the commented import from above."""
        if ftype == np.float32:
            self.C_FTYPE = 'float'
            self.C_PRECISION_DEF = 'SINGLE_PRECISION'
            self.FTYPE = ftype
            self.C_CHANGETYPE = 'int'
            sys.stderr.write("Histogramming is set to single precision (FP32) "
                             "mode.\n\n")
        elif ftype == np.float64:
            self.C_FTYPE = 'double'
            self.C_PRECISION_DEF = 'DOUBLE_PRECISION'
            self.FTYPE = ftype
            self.C_CHANGETYPE = 'unsigned long long int'
            sys.stderr.write("Histogramming is set to double precision (FP64) "
                             "mode.\n\n")
        else:
            raise ValueError('FTYPE must be one of `np.float32` or `np.float64`'
                             '. Got %s instead.' %ftype)


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        #self.clear()
        return


def test_GPUHist():
    """A small test which calculates a histogram"""

if __name__ == '__main__':
    test_GPUHist()
