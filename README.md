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
You are going to need to have the following requirements installed in order to use `gpu_hist.py`:
 * [CUDA](https://developer.nvidia.com/cuda-zone) -- version > 5 recommended; on Ubuntu install via `nvidia-cuda-toolkit`
 * [PyCUDA](https://developer.nvidia.com/pycuda) -- install via pip
 * [NumPy](http://www.numpy.org/) -- install via pip

If you would like to create benchmark tests with `main.py` or plot some histograms, you need:
 * [matplotlib](http://matplotlib.org/) -- install via pip
 * [pandas](http://pandas.pydata.org/) -- install via pip

For further information please visit the [Wiki](https://github.com/PolygonAndPixel/histogramming/wiki)
