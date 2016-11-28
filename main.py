# authors: M. Hieronymus (mhierony@students.uni-mainz.de)
# date:    November 2016
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gpu_hist

FTYPE = np.float64

def create_array(n_elements, n_dimensions):
    """Create an array with values between -360 and 360 (could be any other
    range too)"""
    return np.array(720*np.random.random((n_dimensions, n_elements))-360,
            dtype=FTYPE)


def create_weights(n_elements, n_dimensions):
    #TODO: Check if weights should be normalized
    return np.random.random((n_dimensions, n_elements))


#TODO: Implement this
# Currently only 1D and wrong
def plot_histogram(histogram, edges, outdir, name, no_of_bins):
    """Plots the histogram into specified directory. If the path does not exist
    then it will be created.

    Parameters
    ----------
    histogram : array
    edges : array
    outdir : path
    name : string
    no_of_bins : int (length of edges if edges is given)
    """

    path = [outdir]
    mkdir(os.path.join(*path), warn=False)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 0.35
    if edges is None:
        edges = np.arange(0, 1, (1/no_of_bins))

    rects = ax.bar(edges, histogram, width)
    ax.set_xticks(edges + width)
    xtickNames = ax.set_xticklabels(edges)
    plt.save(outdir+"/"+name)



# TODO: Include timer
# Add all tests
# Create plots of histograms
# Create better/other random arrays?
# Finish gpu_hist.py
# Finish first naive CUDA-code
if __name__ == '__main__':
    """
    This is based on
    https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
    http://parse.ele.tue.nl/system/attachments/10/original/GPGPU_High%20Performance%20Predictable%20Histogramming%20on%20GPUs.pdf?1314781744
    https://isaac.gelado.cat/sites/isaac.gelado.cat/files/publications/samos_2013_histogramming.pdf
    """
    # Do cool stuff
    parser = ArgumentParser(
    description=
            '''Run several tests for histogramming with a GPU.''',
    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--full', action='store_true',
            help=
            '''Full test with comparison of numpy's histogramdd and GPU code
            with single and double precision and the GPU code with shared and
            global memory.''')
    parser.add_argument('--GPU_shared', action='store_true',
            help=
            '''Use GPU code with shared memory. If --GPU_both is set, then
            --GPU_shared will be ignored.''')
    parser.add_argument('--GPU_global', action='store_true',
            help=
            '''Use GPU code with global memory. If --GPU_both is set, then
            --GPU_global will be ignored.''')
    parser.add_argument('--GPU_both', action='store_true',
            help=
            '''Use GPU code with shared memory and global memory and compare
            both.''')
    parser.add_argument('--numpy', action='store_true',
            help=
            '''Use numpy's histogramdd.''')
    parser.add_argument('--all_precisions', action='store_true',
            help=
            '''Run all specified tests with double and single precision.''')
    parser.add_argument('-s', '--single_precision', action='store_true', help=
            '''Use single precision. If it is not set, use double precision.
            If --all_precisions is used, then -s will be ignored.''')
    parser.add_argument('-d', '--data', type=int, required=False,
            default=256*256, help=
            '''Define the number of elements in each dimension for the input
            data.''')
    parser.add_argument('--dimension_data', type=int, required=False, default=1,
            help=
            '''Define the dimensions for the input data.''')
    parser.add_argument('-b', '--bins', type=int, required=False, default=256,
            help=
            '''Choose the number of bins for each dimension''')
    parser.add_argument('--dimension_bins', type=int, required=False, default=1,
            help=
            '''Define the dimensions for the histogram.''')
    parser.add_argument('-w', '--weights', action='store_true',
            help=
            '''(Randomized) weights will be used on the histogram.''')
    parser.add_argument('--outdir', metavar='DIR', type=str,
            help=
            '''Store all output plots to this directory. If
            they don't exist, the script will make them, including
            all subdirectories. If none is supplied no plots will
            be saved.''')
    args = parser.parse_args()

    if args.single_precision and not args.all_precisions and not args.full:
        FTYPE = np.float32
    weights = None
    if args.weights:
        weights = create_weights()
    if args.full:
        input_data = create_array(args.data, args.dimension_data)
        # First with double precision
        with gpu_hist.GPUHist(FTYPE=FTYPE) as histogrammer:
            histogram_d_gpu_shared = histogrammer.get_hist(shared=True)
            histogram_d_gpu_global = histogrammer.get_hist(shared=False)
        histogram_d_numpy, edges_d = np.histogramdd(input_data, bins=args.bins,
                weights=weights)
        # Next with single precision
        FTYPE = np.float32
        with gpu_hist.GPUHist(FTYPE=FTYPE) as histogrammer:
            histogram_s_gpu_shared = histogrammer.get_hist(shared=True)
            histogram_s_gpu_global = histogrammer.get_hist(shared=False)
        histogram_s_numpy, edges_s = np.histogramdd(input_data, bins=args.bins,
                weights=weights)
        if(outdir != None):
            plot_histogram(histogram_d_gpu_shared, None, args.outdir,
                    "GPU shared memory, double", args.bins)
            plot_histogram(histogram_d_gpu_global, None, args.outdir,
                    "GPU global memory, double", args.bins)
            plot_histogram(histogram_d_numpy, edges_d, args.outdir,
                    "Numpy, double", args.bins)
            plot_histogram(histogram_s_gpu_shared, None, args.outdir,
                    "GPU shared memory, single", args.bins)
            plot_histogram(histogram_s_gpu_global, None, args.outdir,
                    "GPU global memory, single", args.bins)
            plot_histogram(histogram_s_numpy, edges_s, args.outdir,
                    "Numpy, single", args.bins)
        sys.exit()
    if args.GPU_both:
        # if not args.all_precisions and args.single_precision then this is
        # single precision. Hence the missing "d" or "s" in the name.
        with gpu_hist.GPUHist(FTYPE=FTYPE) as histogrammer:
            histogram_gpu_shared = histogrammer.get_hist(shared=True)
            histogram_gpu_global = histogrammer.get_hist(shared=False)
        if args.all_precisions:
            FTYPE = np.float32
            with gpu_hist.GPUHist(FTYPE=FTYPE) as histogrammer:
                histogram_s_gpu_shared = histogrammer.get_hist(shared=True)
                histogram_s_gpu_global = histogrammer.get_hist(shared=False)
            plot_histogram(histogram_gpu_shared, None, args.outdir,
                    "GPU shared memory, double", args.bins)
            plot_histogram(histogram_gpu_global, None, args.outdir,
                    "GPU global memory, double", args.bins)
            plot_histogram(histogram_s_gpu_shared, None, args.outdir,
                    "GPU shared memory, single", args.bins)
            plot_histogram(histogram_s_gpu_global, None, args.outdir,
                    "GPU global memory, single", args.bins)
        else:
            name = ""
            if args.single_precision:
                name = "single"
            else:
                name = "double"
            plot_histogram(histogram_gpu_shared, None, args.outdir,
                    "GPU shared memory, " + name, args.bins)
            plot_histogram(histogram_gpu_global, None, args.outdir,
                    "GPU global memory, " + name, args.bins)

    if args.GPU_shared and not args.GPU_both:
        with gpu_hist.GPUHist(FTYPE=FTYPE) as histogrammer:
            histogram_gpu_shared = histogrammer.get_hist(shared=True)
        if args.all_precisions:
            FTYPE = np.float32
            with gpu_hist.GPUHist(FTYPE=FTYPE) as histogrammer:
                histogram_s_gpu_shared = histogrammer.get_hist(shared=True)
            plot_histogram(histogram_gpu_shared, None, args.outdir,
                    "GPU shared memory, double", args.bins)
            plot_histogram(histogram_s_gpu_shared, None, args.outdir,
                    "GPU shared memory, single", args.bins)
        else:
            name = ""
            if args.single_precision:
                name = "single"
            else:
                name = "double"
            plot_histogram(histogram_gpu_shared, None, args.outdir,
                    "GPU shared memory, " + name, args.bins)

    if args.GPU_global and not args.GPU_both:
        with gpu_hist.GPUHist(FTYPE=FTYPE, no_of_dimensions=args.dimension_bins,
                no_of_bins=args.bins) as histogrammer:
            histogram_gpu_global = histogrammer.get_hist(shared=False)
        if args.all_precisions:
            FTYPE = np.float32
            with gpu_hist.GPUHist(FTYPE=FTYPE) as histogrammer:
                histogram_s_gpu_global = histogrammer.get_hist(shared=False)
            plot_histogram(histogram_gpu_global, None, args.outdir,
                    "GPU global memory, double", args.bins)
            plot_histogram(histogram_s_gpu_global, None, args.outdir,
                    "GPU global memory, single", args.bins)
        else:
            name = ""
            if args.single_precision:
                name = "single"
            else:
                name = "double"
            plot_histogram(histogram_gpu_global, None, args.outdir,
                    "GPU global memory, " + name, args.bins)
