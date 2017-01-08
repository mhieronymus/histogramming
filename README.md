# Histogramming
My goal is to create a generic and fast GPU-based histogramming which can be
used as a Python module. This is based on [an implementation by NVIDIA](https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/).
This module should be compatible with PISA (PINGU Simulation and Analysis).
All goals in detail:

 * Compare timings to numpy's `histogramdd` and be faster
 * Work on arrays that are already on GPU (requirement from PISA)
 * Ability to move additional arrays to GPU (requirement from PISA)
 * Should be used as a Python module
 * Identify the GPU and open threads accordingly (and use best kernel)
 * Create generic histograms for generic data (N dimensional data, different bins, M dimensional histogram etc.)
 * Implement a test method to compare the results
 * Support single and double precision (requirement from PISA)

## Prerequisites
PyCUDA, CUDA etc.

## Usage
Simply type `python main.py` and use some of the following options:

 * `--full`: Full test with comparison of numpy's histogramdd and GPU code
 with single and double precision and the GPU code with shared and global memory.
 * `--GPU_shared`: Use GPU code with shared memory. If --GPU_both is set, then
  --GPU_shared will be ignored.
 * `--GPU_global`: Use GPU code with global memory. If --GPU_both is set, then
  --GPU_global will be ignored.
 * `--GPU_both`: Use GPU code with shared memory and global memory and compare both.        
 * `--CPU`: Use numpy's histogramdd.        
 * `--all_precisions`: Run all specified tests with double and single precision.
 * `-s`, `--single_precision`: Use single precision. If it is not set, use double precision.
 If --all_precisions is used, then -s will be ignored.
 * `-d`, `--data`: Define the number of elements in each dimension for the input data.  
 * `--dimension`: Define the number of dimensions for the input data and the histogram.
 * `-b`, `--bins`: Choose the number of bins for each dimension    
 * `-w`, `--weights`: (Randomized) weights will be used on the histogram.  
 * `--use_given_edges`: Use calculated edges instead of calculating edges during histogramming.
 * `--outdir`: Store all output plots to this directory. If they don't exist,
 the script will make them, including all subdirectories.
 If none is supplied no plots will be saved.

## Input
Currently some arbitrary values between -360 and 360 are generated.

## Output
Currently only a small comparison between numpy's implementation and the GPU
version are shown (if `--outdir` is set and both tests are used).

## Histogramming
Before a thread reads a value it is not known in which bin it belongs which can
cause collisions with other threads. Therefore atomic operations must be used
to construct a histogram. Those operations are fast since Kepler and
shared memory atomics are fast since Maxwell.

The implementation provided by NVIDIA uses a two-phase parallel histogram
algorithm where local histograms are computed using atomics in phase 1 and
then merged in phase 2 (each phase gets one kernel).

### Phase 1
Each CUDA thread block processes a region of the input data which reduces
conflicts if many values belong to one bin. NVIDIA provides two implementations
for phase 1:
#### Use global memory
This implementation stores per-block local histograms in global memory.
One should use this approach if shared memory atomics are slow (e.g. GPUs
older than Maxwell). In this case the cost for shared memory atomics is much
higher than the cost for using global memory.

#### Use shared memory
This implementation stores per-block local histograms in shared memory.
Using global memory is slow but histogramming needs lots of shared memory
atomics so this should only be used for efficient GPUs (Maxwell and newer).
In the end of this kernel all local histograms are copied to global memory once.

### Phase 2
All histograms are merged together. This kernel is independent from the chosen
kernel of phase 1 since all local histograms are in global memory.

### Using N-dimensional input data
The implementation by NVIDIA is shown on 2D-data but it can be extended to N
dimensions easily.

## Results
NVIDIA's implementation shows good results with high entropy (e.g. perfect
distribution of the data and only a few atomic conflicts) with shared memory
and a GTX Titan (Kepler architecture). In all other cases (and in all cases
with a GTX Titan X) the kernel with shared memory performs better.

## Future Improvements
Using RLE encoding for the bin identifiers. For large multidimensional
histograms: Represent them as a linear histogram and use a multi-pass approach
similar to radix sort for highly conflict-laden party
(if they don't fit into shared memory).
With a high entropy in input data one can just use global memory.
