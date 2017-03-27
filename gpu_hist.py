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

        kernel_code = open("gpu_hist/histogram_atomics.cu", "r").read() %dict(
            c_precision_def=self.C_PRECISION_DEF,
            c_ftype=self.C_FTYPE,
            c_itype=self.C_ITYPE,
            c_histotype=self.C_HIST_TYPE,
            c_changetype=self.C_CHANGETYPE
        )
        include_dirs = ['/gpu_hist']
        # keep for compiler output, no_extern_c: allow name manling
        # Add -g for debug mode
        module = SourceModule(kernel_code, keep=True,
                              options=['--compiler-options', '-Wall'],
                              include_dirs=include_dirs, no_extern_c=False)
        self.max_min_reduce = module.get_function("max_min_reduce")
        self.max_min_reduce2 = module.get_function("max_min_reduce2")

        self.hist_gmem = module.get_function("histogram_gmem_atomics")
        self.hist_gmem_given_edges = module.get_function("histogram_gmem_atomics_with_edges")
        self.hist_gmem_weights = module.get_function("histogram_gmem_atomics_weights")
        self.hist_gmem_given_edges_weights = module.get_function("histogram_gmem_atomics_with_edges_weights")
        # Following functions use different shape of input arrays.
        self.hist_gmem2 = module.get_function("histogram_gmem_atomics2")
        self.hist_gmem_given_edges2 = module.get_function("histogram_gmem_atomics_with_edges2")
        self.hist_gmem_weights2 = module.get_function("histogram_gmem_atomics_weights2")
        self.hist_gmem_given_edges_weights2 = module.get_function("histogram_gmem_atomics_with_edges_weights2")

        self.hist_smem = module.get_function("histogram_smem_atomics")
        self.hist_smem_given_edges = module.get_function("histogram_smem_atomics_with_edges")
        self.hist_smem_weights = module.get_function("histogram_smem_atomics_weights")
        self.hist_smem_given_edges_weights = module.get_function("histogram_smem_atomics_with_edges_weights")
        # Following functions use different shape of input arrays
        self.hist_smem2 = module.get_function("histogram_smem_atomics2")
        self.hist_smem_given_edges2 = module.get_function("histogram_smem_atomics_with_edges2")
        self.hist_smem_weights2 = module.get_function("histogram_smem_atomics_weights2")
        self.hist_smem_given_edges_weights2 = module.get_function("histogram_smem_atomics_with_edges_weights2")

        self.hist_accum = module.get_function("histogram_final_accum")
        self.hist_accum_weights = module.get_function("histogram_final_accum_weights")

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

        self.d_hist = None
        self.d_edges_in = None
        self.edges = None
        self.flattened = False

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
        """Free the edges and the histogram on the GPU."""
        if self.d_hist is not None:
            self.d_hist.free() # Should not be needed.
        if self.d_edges_in is not None:
            self.d_edges_in.free()


    def set_bins(self, bins, dims=1):
        """Copy the bins to the GPU and reuse them. This is highly recommended
        if you are using the same bins multiple times.

        Parameters
        ----------
        bins: array or list of arrays or list or int
              An of arrays describing the bin edges along each dimension or
              a list of arrays describing the bin edges along each dimension or
              a list describing the number of bins for each dimension or
              the number of bins for all dimensions.
        dims: int, optional
              Give the number of dimensions for your bins.

        Returns
        -------
        d_no_of_bins: cuda.DeviceAllocation
                      Pointer to an array with the number of bins in each
                      dimension on the GPU. Use this in `get_hist` in
                      `bins`.

        """
        self.clear()
        # Check if number of bins for all dimensions is given or
        # if number of bins for each dimension is given or
        # if the edges for each dimension are given
        if isinstance(bins, int):
            # Use equally spaced bins in all dimensions
            self.n_flat_bins = self.ITYPE(self.ITYPE(bins) ** dims)
            self.no_of_bins = [self.ITYPE(bins) for _ in xrange(dims)]
            self.no_of_bins = np.asarray(self.no_of_bins)
            d_no_of_bins = cuda.mem_alloc(self.no_of_bins.nbytes)
            cuda.memcpy_htod(d_no_of_bins, self.no_of_bins)
        elif not isinstance(bins[0], list) and not isinstance(bins[0], np.ndarray):
            # Use different amounts of bins in each dimension
            self.n_flat_bins = 1
            self.no_of_bins = []
            for b in bins:
                self.n_flat_bins = self.n_flat_bins * b
                self.no_of_bins.append(self.ITYPE(b))
            self.no_of_bins = np.asarray(self.no_of_bins)
            self.d_no_of_bins = cuda.mem_alloc(self.no_of_bins.nbytes)
            cuda.memcpy_htod(d_no_of_bins, self.no_of_bins)
        else:
            # Use given edges
            self.n_flat_bins = 1
            self.no_of_bins = []
            for b in bins:
                self.n_flat_bins = self.n_flat_bins * (len(b) - 1)
                self.no_of_bins.append(self.ITYPE(len(b)-1))
            self.no_of_bins = np.asarray(self.no_of_bins)
            d_no_of_bins = cuda.mem_alloc(self.no_of_bins.nbytes)
            cuda.memcpy_htod(d_no_of_bins, self.no_of_bins)
            self.n_flat_bins = self.ITYPE(self.n_flat_bins)
            if isinstance(bins, list):
                # Different amount of bins for each dimension. Therefore flatten
                # the list before casting to array
                self.edges = np.asarray([item for sublist in bins for item in sublist])
                self.flattened = True
            else:
                self.edges = bins
            self.d_edges_in = cuda.mem_alloc(self.edges.nbytes)
            cuda.memcpy_htod(self.d_edges_in, self.edges)
        return d_no_of_bins


    def get_hist(self, sample, shared=True, bins=10, normed=False,
                 weights=None, dims=1, number_of_events=0):
        """Retrive histogram with given events.

        Parameters
        ----------
        sample: array or cuda.DeviceAllocation or list of cuda.DeviceAllocation
                The events to be histogrammed. It must be an (N,D) array with N
                elements and D dimensions or a device array with N x D entries
                in the desired precision (double or float). A (D) list of (N)
                device arrays where each array holds one dimension works for 3
                or less dimensions. The latter should be faster than the other
                inputs.

        shared: bool, optional
                If False, global memory will be used for computing the
                histogram. This should be done for debug purpose or if you want
                to compare the speed between the global and shared memory
                approach.

        bins: array or list of arrays or list or int, optional
              An of arrays describing the bin edges along each dimension or
              a list of arrays describing the bin edges along each dimension or
              a list describing the number of bins for each dimension or
              the number of bins for all dimensions.

        normed: bool, optional
                If False, returns the number of samples in each bin. If True,
                return the bin density bin_count / sample_count / bin_volume.
                This is not implemented yet!

        weights: (N,) array, optional
                 An array of values w_i for each sample (x_i, y_i, z_i, ...).
                 Weights are normalized to 1 if normed is True (not implemented
                 yet). If normed is False, the values of the returned histogram
                 are equal to the sum of the weights belonging to the samples
                 falling into each bin.

        dims: int, required if events are cuda.DeviceAllocation
              Give the number of dimensions for your events.

        number_of_events: int, required if events are cuda.DeviceAllocation or list of cuda.DeviceAllocation
                          Give the number of events for your device_array.

        Returns
        -------
        hist: ndarray
              The multidimensional histogram of the events.

        edges: list
               A list of D arrays describing the bin edges for each dimension.
        """
        t0 = time.time()
        list_of_device_arrays = False
        # If we got weights, we need to change the type of the histogram from
        # integer to FTYPE
        if not weights is None:
            self.HIST_TYPE = self.FTYPE
            self.C_HIST_TYPE = self.C_FTYPE

        if isinstance(sample, cuda.DeviceAllocation):
            if number_of_events > 0:
                n_dims = dims
                n_events = number_of_events
            else:
                raise ValueError("If you use a device array as input, you have "
                                 "to specify the number of events in your input "
                                 "and the number of dimensions (default is 1 "
                                 "for dimensions).\n\n")
        elif isinstance(sample, list) and isinstance(sample[0], cuda.DeviceAllocation):
            if number_of_events > 0:
                n_dims = len(sample)
                n_events = number_of_events
                list_of_device_arrays = True
            else:
                raise ValueError("If you use a list of device arrays as input, "
                                 "you have to specify the number of events in "
                                 "your input.\n\n")
        else:
            try:
                n_events, n_dims = sample.shape
            except (AttributeError, ValueError):
                sample = np.atleast_2d(sample).T
                n_events, n_dims = sample.shape
            n_dims = self.ITYPE(n_dims)

        d_max_in = None
        d_min_in = None

        sizeof_hist_t = np.dtype(self.HIST_TYPE).itemsize
        sizeof_c_ftype = np.dtype(self.C_FTYPE).itemsize
        sizeof_float_t = np.dtype(self.FTYPE).itemsize

        if isinstance(bins, cuda.DeviceAllocation):
            d_no_of_bins = bins
        else:
            d_no_of_bins = self.set_bins(bins)

        self.set_block_dims(sizeof_c_ftype, n_dims, False)
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
        elif isinstance(sample[0], cuda.DeviceAllocation):
            d_sample = [s for s in sample]
            # Dirty hack: Lists of device arrays are supported for 3 dimensions
            # If less arrays are given we allocate dummy arrays
            for x in xrange(n_dims, 3):
                d_sample.append(cuda.mem_alloc(8))
        else:
            d_sample = cuda.mem_alloc(sample.nbytes)
            cuda.memcpy_htod(d_sample, sample)
        if isinstance(weights, cuda.DeviceAllocation):
            d_weights = weights
        elif not weights is None:
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
            if self.edges is None:
                d_max_in = cuda.mem_alloc(n_dims * sizeof_float_t)
                d_min_in = cuda.mem_alloc(n_dims * sizeof_float_t)
                self.set_block_dims(sizeof_c_ftype, n_dims, True)
                if list_of_device_arrays:
                    self.max_min_reduce2(d_sample[0],
                                        self.ITYPE(n_events), d_sample[1],
                                        d_sample[2],
                                        self.ITYPE(n_dims), d_max_in, d_min_in,
                                        block=self.block_dim, grid=self.grid_dim,
                                        shared=self.shared)
                else:
                    self.max_min_reduce(d_sample,
                                        self.ITYPE(n_events),
                                        self.ITYPE(n_dims), d_max_in, d_min_in,
                                        block=self.block_dim, grid=self.grid_dim,
                                        shared=self.shared)
                self.shared = (self.n_flat_bins * sizeof_hist_t)
                self.set_block_dims(sizeof_c_ftype, n_dims, False)
                if weights is None:
                    # Calculate local histograms on shared memory on device
                    if list_of_device_arrays:
                        self.hist_smem2(d_sample[0],
                                       self.ITYPE(n_events),
                                       d_sample[1], d_sample[2],
                                       self.ITYPE(n_dims),
                                       d_no_of_bins,
                                       self.ITYPE(self.n_flat_bins),
                                       d_tmp_hist, d_max_in, d_min_in,
                                       block=self.block_dim, grid=self.grid_dim,
                                       shared=self.shared)
                    else:
                        self.hist_smem(d_sample,
                                       self.ITYPE(n_events*n_dims),
                                       self.ITYPE(n_dims),
                                       d_no_of_bins,
                                       self.ITYPE(self.n_flat_bins),
                                       d_tmp_hist, d_max_in, d_min_in,
                                       block=self.block_dim, grid=self.grid_dim,
                                       shared=self.shared)
                else: # with weights
                    # Calculate local histograms with weights
                    if list_of_device_arrays:
                        self.hist_smem_weights2(d_sample[0],
                                               self.ITYPE(n_events),
                                               d_sample[1], d_sample[2],
                                               self.ITYPE(n_dims),
                                               d_no_of_bins,
                                               self.ITYPE(self.n_flat_bins),
                                               d_tmp_hist, d_max_in, d_min_in,
                                               d_weights,
                                               block=self.block_dim,
                                               grid=self.grid_dim,
                                               shared=self.shared)
                    else:
                        self.hist_smem_weights(d_sample,
                                               self.ITYPE(n_events*n_dims),
                                               self.ITYPE(n_dims),
                                               d_no_of_bins,
                                               self.ITYPE(self.n_flat_bins),
                                               d_tmp_hist, d_max_in, d_min_in,
                                               d_weights,
                                               block=self.block_dim,
                                               grid=self.grid_dim,
                                               shared=self.shared)
            else:
                self.shared = (self.n_flat_bins * sizeof_hist_t)
                if self.d_edges_in is None:
                    self.d_edges_in = cuda.mem_alloc(self.edges.nbytes)
                    cuda.memcpy_htod(self.d_edges_in, self.edges)
                if weights is None:
                    if list_of_device_arrays:
                        self.hist_smem_given_edges2(d_sample[0],
                                                   self.ITYPE(n_events),
                                                   d_sample[1], d_sample[2],
                                                   self.ITYPE(n_dims),
                                                   d_no_of_bins,
                                                   self.ITYPE(self.n_flat_bins),
                                                   d_tmp_hist, self.d_edges_in,
                                                   block=self.block_dim,
                                                   grid=self.grid_dim,
                                                   shared=self.shared)
                    else:
                        self.hist_smem_given_edges(d_sample,
                                                   self.ITYPE(n_events*n_dims),
                                                   self.ITYPE(n_dims),
                                                   d_no_of_bins,
                                                   self.ITYPE(self.n_flat_bins),
                                                   d_tmp_hist, self.d_edges_in,
                                                   block=self.block_dim,
                                                   grid=self.grid_dim,
                                                   shared=self.shared)
                else:
                   # Calculate local histograms with edges and weights
                   if list_of_device_arrays:
                       self.hist_smem_given_edges_weights2(d_sample[0],
                                                          self.ITYPE(n_events),
                                                          d_sample[1], d_sample[2],
                                                          self.ITYPE(n_dims),
                                                          d_no_of_bins,
                                                          self.ITYPE(self.n_flat_bins),
                                                          d_tmp_hist, self.d_edges_in,
                                                          d_weights,
                                                          block=self.block_dim,
                                                          grid=self.grid_dim,
                                                          shared=self.shared)
                   else:
                       self.hist_smem_given_edges_weights(d_sample,
                                                          self.ITYPE(n_events*n_dims),
                                                          self.ITYPE(n_dims),
                                                          d_no_of_bins,
                                                          self.ITYPE(self.n_flat_bins),
                                                          d_tmp_hist, self.d_edges_in,
                                                          d_weights,
                                                          block=self.block_dim,
                                                          grid=self.grid_dim,
                                                          shared=self.shared)
        else: # global memory
            # Calculate edges by yourself if no edges are given
            if self.edges is None:
                d_max_in = cuda.mem_alloc(n_dims * sizeof_float_t)
                d_min_in = cuda.mem_alloc(n_dims * sizeof_float_t)
                self.set_block_dims(sizeof_c_ftype, n_dims, True)
                if list_of_device_arrays:
                    self.max_min_reduce2(d_sample[0],
                                        self.ITYPE(n_events),
                                        d_sample[1], d_sample[2],
                                        self.ITYPE(n_dims), d_max_in, d_min_in,
                                        block=self.block_dim, grid=self.grid_dim,
                                        shared=self.shared)
                else:
                    self.max_min_reduce(d_sample,
                                        self.ITYPE(n_events),
                                        self.ITYPE(n_dims), d_max_in, d_min_in,
                                        block=self.block_dim, grid=self.grid_dim,
                                        shared=self.shared)
                self.set_block_dims(sizeof_c_ftype, n_dims, False)
                if weights is None:
                    if list_of_device_arrays:
                        self.hist_gmem2(d_sample[0],
                                       self.ITYPE(n_events),
                                       d_sample[1], d_sample[2],
                                       self.ITYPE(n_dims),
                                       d_no_of_bins,
                                       self.ITYPE(self.n_flat_bins),
                                       d_tmp_hist, d_max_in, d_min_in,
                                       block=self.block_dim, grid=self.grid_dim)
                    else:
                        self.hist_gmem(d_sample,
                                       self.ITYPE(n_events*n_dims),
                                       self.ITYPE(n_dims),
                                       d_no_of_bins,
                                       self.HIST_TYPE(self.n_flat_bins),
                                       d_tmp_hist, d_max_in, d_min_in,
                                       block=self.block_dim, grid=self.grid_dim)
                else:
                    # Calculate global histograms with weights
                    if list_of_device_arrays:
                        self.hist_gmem_weights2(d_sample[0],
                                               self.ITYPE(n_events),
                                               d_sample[1], d_sample[2],
                                               self.ITYPE(n_dims),
                                               d_no_of_bins,
                                               self.ITYPE(self.n_flat_bins),
                                               d_tmp_hist, d_max_in, d_min_in,
                                               d_weights,
                                               block=self.block_dim,
                                               grid=self.grid_dim)
                    else:
                        self.hist_gmem_weights(d_sample,
                                               self.ITYPE(n_events*n_dims),
                                               self.ITYPE(n_dims),
                                               d_no_of_bins,
                                               self.ITYPE(self.n_flat_bins),
                                               d_tmp_hist, d_max_in, d_min_in,
                                               d_weights,
                                               block=self.block_dim,
                                               grid=self.grid_dim)
            else:
                if self.d_edges_in is None:
                    self.d_edges_in = cuda.mem_alloc(self.edges.nbytes)
                    cuda.memcpy_htod(self.d_edges_in, self.edges)
                if weights is None:
                    if list_of_device_arrays:
                        self.hist_gmem_given_edges2(d_sample[0],
                                                   self.ITYPE(n_events),
                                                   d_sample[1], d_sampl[2],
                                                   self.ITYPE(n_dims),
                                                   d_no_of_bins,
                                                   self.ITYPE(self.n_flat_bins),
                                                   d_tmp_hist, self.d_edges_in,
                                                   block=self.block_dim,
                                                   grid=self.grid_dim)
                    else:
                        self.hist_gmem_given_edges(d_sample,
                                                   self.ITYPE(n_events*n_dims),
                                                   self.ITYPE(n_dims),
                                                   d_no_of_bins,
                                                   self.ITYPE(self.n_flat_bins),
                                                   d_tmp_hist, self.d_edges_in,
                                                   block=self.block_dim,
                                                   grid=self.grid_dim)
                else:
                    # Calculate global histograms with edges and weights
                    if list_of_device_arrays:
                        self.hist_gmem_given_edges_weights2(d_sample[0],
                                                           self.ITYPE(n_events),
                                                           d_sample[1], d_sample[2],
                                                           self.ITYPE(n_dims),
                                                           d_no_of_bins,
                                                           self.ITYPE(self.n_flat_bins),
                                                           d_tmp_hist, self.d_edges_in,
                                                           d_weights,
                                                           block=self.block_dim,
                                                           grid=self.grid_dim)
                    else:
                        self.hist_gmem_given_edges_weights(d_sample,
                                                           self.ITYPE(n_events*n_dims),
                                                           self.ITYPE(n_dims),
                                                           d_no_of_bins,
                                                           self.ITYPE(self.n_flat_bins),
                                                           d_tmp_hist, self.d_edges_in,
                                                           d_weights,
                                                           block=self.block_dim,
                                                           grid=self.grid_dim)

        if weights is None:
            self.hist_accum(d_tmp_hist, self.ITYPE(self.grid_dim[0]), self.d_hist,
                            self.ITYPE(self.n_flat_bins),
                            block=self.block_dim, grid=self.grid_dim)
        else:
            self.hist_accum_weights(d_tmp_hist, self.ITYPE(self.grid_dim[0]),
                                    self.d_hist, self.ITYPE(self.n_flat_bins),
                                    block=self.block_dim, grid=self.grid_dim)
        # Copy the array back and make the right shape
        cuda.memcpy_dtoh(self.hist, self.d_hist)
        histo_shape = ()
        for d in range(0, n_dims):
            histo_shape += (self.no_of_bins[d], )
        self.hist = np.reshape(self.hist, histo_shape)

        if self.edges is None:
            # Calculate the found edges
            max_in = np.zeros(n_dims, dtype=self.FTYPE)
            min_in = np.zeros(n_dims, dtype=self.FTYPE)
            cuda.memcpy_dtoh(max_in, d_max_in)
            cuda.memcpy_dtoh(min_in, d_min_in)
            self.edges = []
            # Create some nice edges
            for d in range(0, n_dims):
                try:
                    edges_d = np.linspace(min_in[d], max_in[d],
                                          self.no_of_bins[d]+1,
                                          dtype=self.FTYPE)
                except ValueError:
                    print min_in[d], max_in[d], self.no_of_bins[d], self.FTYPE
                    raise
                self.edges.append(edges_d)

        self.d_hist.free()
        d_tmp_hist.free()

        if not isinstance(sample, cuda.DeviceAllocation) and not isinstance(d_sample, list):
            d_sample.free()
        if not isinstance(weights, cuda.DeviceAllocation) and not weights is None:
            d_weights.free()
        if d_max_in is not None:
            d_max_in.free()
        if d_min_in is not None:
            d_min_in.free()

        # Check if edges had to be flattened before:
        if self.flattened:
            self.edges = bins

        self.calc_time = time.time() - t0

        return self.hist, self.edges


    def set_block_dims(self, sizeof_c_ftype, n_dims, max_min_reduction):
        """Set block dimensions according to the given dimensions and the
        application. We use a one-dimensional block and grid.
        We use as many threads per block as possible but we are limited by the
        shared memory.

        Parameters
        ----------
        sizeof_c_ftype: The itemsize of C_FTYPE
        n_dims: The dimensions of the sample data
        max_min_reduction: True if dimensions should be set for the reduction
        """
        if max_min_reduction:
            self.block_dim = (self.max_threads_per_block, 1, 1)
        else:
            no_of_threads = (self.shared_memory / sizeof_c_ftype * 2)
            if no_of_threads > self.max_threads_per_block:
                overflow = self.max_threads_per_block%n_dims
                self.block_dim = (self.max_threads_per_block-overflow, 1, 1)
            else:
                overflow = no_of_threads%n_dims
                self.block_dim = (no_of_threads-overflow, 1, 1)


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
    all_dims = [1, 2, 3]
    all_elements = np.logspace(5, 7, 3)
    all_bins = np.logspace(1, 4, 4)
    all_ftypes = [np.float32, np.float64]
    all_device_samples = [False, True]
    all_given_edges = [False, True]
    gpu_attributes = cuda.Device(0).get_attributes()
    max_threads_per_block = gpu_attributes.get(
        cuda.device_attribute.MAX_THREADS_PER_BLOCK
    )
    for n_dims, n_elements, n_bins, ftype, device_samples, given_edges in product(
            all_dims, all_elements, all_bins, all_ftypes,
            all_device_samples, all_given_edges):
        n_elements = int(n_elements)
        n_bins = int(n_bins)
        # Check if everything fits on the GPU. Continue if it is not the case.
        # One integer is 4 bytes. We need to know how many blocks there are
        # with their own histogram. We also take the samples into account
        # and the edges if they are given and need to be copied.
        dx, mx = divmod(n_elements, max_threads_per_block)
        grid_dim = dx + (mx > 0)
        # local histograms
        n_bytes = n_bins**n_dims*grid_dim*4

        if ftype == np.float32:
            # samples
            n_bytes += n_dims*n_elements*4
            if given_edges:
                n_bytes += 4*n_bins**n_dims
        else:
            # samples
            n_bytes += n_dims*n_elements*8
            if given_edges:
                n_bytes += 8*n_bins**n_dims
        available_memory = cuda.mem_get_info()[0]
        if n_bytes > available_memory:
            continue

        # CPU
        # Create test data inside the loop to avoid caching
        input_data, d_input_data = create_array(
            n_elements=n_elements,
            n_dims=n_dims,
            device_array=False,
            ftype=ftype,
            list_array=args.list_data
        )
        edges = None
        if given_edges:
            edges = create_edges(n_bins=n_bins, n_dims=n_dims,
                                 random=False, ftype=ftype)
        else:
            edges = n_bins

        histogram_numpy, edges_numpy = np.histogramdd(
            input_data, bins=edges, weights=weights
        )
        if isinstance(d_input_data, cuda.DeviceAllocation):
            d_input_data.free()

        # GPU global memory
        with gpu_hist.GPUHist(ftype=ftype) as histogrammer:
            # Create test data inside the loop to avoid caching
            input_data, d_input_data = create_array(
                n_elements=n_elements,
                n_dims=n_dims,
                device_array=device_samples,
                ftype=ftype,
                list_array=args.list_data
            )
            edges = None
            if given_edges:
                edges = create_edges(n_bins=n_bins, n_dims=n_dims,
                                     random=False, ftype=ftype)
            else:
                edges = n_bins

            histogram_gpu_global, edges_gpu_global = histogrammer.get_hist(
                sample=d_input_data, bins=edges, shared=False,
                dims=n_dims, number_of_events=n_elements
            )
            if isinstance(d_input_data, cuda.DeviceAllocation):
                d_input_data.free()

        # GPU shared memory
        tmp_timings = []
        with gpu_hist.GPUHist(ftype=ftype) as histogrammer:
            # Create test data inside the loop to avoid caching
            input_data, d_input_data = create_array(
                n_elements=n_elements,
                n_dims=n_dims,
                device_array=device_samples,
                ftype=ftype,
                list_array=args.list_data
            )
            edges = None
            if given_edges:
                edges = create_edges(n_bins=n_bins, n_dims=n_dims,
                                     random=False, ftype=ftype)
            else:
                edges = n_bins

            histogram_gpu_shared, edges_gpu_shared = histogrammer.get_hist(
                sample=d_input_data, bins=edges, shared=True,
                dims=n_dims, number_of_events=n_elements
            )
            if isinstance(d_input_data, cuda.DeviceAllocation):
                d_input_data.free()
        info_string = 'Comparing outputs with n_elements: %i, input type: %s,' \
                     'dimensions: %i , given_edges: %s, n_bins: %i, ' \
                     'device_samples: %s' (n_elements, ftype, n_dims,
                     given_edges, n_bins, device_samples)
        passed = check_outputs(histo_np=histogram_numpy,
                               histo_global=histogram_gpu_global,
                               histo_shared=histogram_gpu_shared)
        if passed:
            logging.debug(info_string)
            logging.debug('passed test')
        else:
            logging.info(info_string)
            logging.info('Failed test')


def check_outputs(histo_np, histo_global, histo_shared):
    """Compare the given arrays and return True if they are the same and
    false if at least one of them is different than the other ones.
    TODO: Add comparison which returns where the error is."""
    return (histo_np == histo_global and histo_np == histo_shared)


def create_edges(n_bins, n_dims, random=False, seed=0, ftype=FTYPE):
    """Create some random edges given the number of bins for each dimension.
    Used for test_GPUHist."""
    edges = []
    if random:
        np.random.RandomState(seed)
        for dim in range(0, n_dims):
            tmp_bins = rnd.randint(n_bins/2, 3*n_bins/2)
            bin_width = 720.0/tmp_bins
            end_bin = 360.0 + bin_width/10
            edges_d = np.arange(-360.0, end_bin, bin_width, dtype=ftype)
            edges.append(edges_d)
        # Irregular dimensions cannot be casted to arrays.
        return edges
    else:
        for dim in range(0, n_dims):
            bin_width = 720.0/n_bins
            end_bin = 360.0 + bin_width/10
            edges_d = np.arange(-360.0, end_bin, bin_width, dtype=ftype)
            edges.append(edges_d)
    return np.asarray(edges, dtype=ftype)


def create_weights(n_elements, device_array, seed=0, ftype=FTYPE):
    """Create arbitrary weights for the input. Used for test_GPUHist."""
    rand = np.random.RandomState(seed)
    weights = rand.uniform(size=n_elements).astype(ftype)
    if device_array:
        try:
            d_weights = cuda.mem_alloc(weights.nbytes)
            cuda.memcpy_htod(d_weights, weights)
            return weights, d_weights
        except pycuda._driver.MemoryError:
            print "Error at allocating memory"
            available_memory = cuda.mem_get_info()[0]
            print ("You have %d Mbytes memory. Trying to allocate %d"
                   " bytes (%d Mbytes) of memory\n"
                   % (available_memory/(1024*1024), weights.nbytes,
                      weights.nbytes/(1024*1024)))
            return weights, weights
    else:
        return weights, weights


def create_array(n_elements, n_dims, device_array, list_array, seed=0, ftype=FTYPE):
    """Create an arbitrary array for test_GPUHist."""
    assert n_elements > 0
    assert n_dims > 0
    rand = np.random.RandomState(seed)
    values = rand.normal(size=(n_elements, n_dims)).astype(ftype)
    if device_array or (list_array and n_dims > 3):
        try:
            d_values = cuda.mem_alloc(values.nbytes)
            cuda.memcpy_htod(d_values, values)
            return values, d_values
        except pycuda._driver.MemoryError:
            print "Error at allocating memory"
            available_memory = cuda.mem_get_info()[0]
            print ("You have %d Mbytes memory. Trying to allocate %d"
                   " bytes (%d Mbytes) of memory\n"
                   % (available_memory/(1024*1024), values.nbytes,
                      values.nbytes/(1024*1024)))
            return values, values
    elif list_array and n_dims < 4:
        try:
            # We need a different shape here: Each array in a list shall
            # contain one dimension of all data.
            d_values = []
            for i in xrange(n_dims):
                tmp_values = np.asarray([v[i] for v in values])
                d_values.append(cuda.mem_alloc(tmp_values.nbytes))
                cuda.memcpy_htod(d_values[i], tmp_values)
            return values, d_values
        except pycuda._driver.MemoryError:
            print "Error at allocating memory"
            available_memory = cuda.mem_get_info()[0]
            print ("You have %d Mbytes memory. Trying to allocate %d"
                   " bytes (%d Mbytes) of memory\n"
                   % (available_memory/(1024*1024), values.nbytes,
                      values.nbytes/(1024*1024)))
            return values, values
    else:
        return values, values

if __name__ == '__main__':
    test_GPUHist()
