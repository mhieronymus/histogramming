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

## Input

## Output

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
