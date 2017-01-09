# authors: M. Hieronymus (mhierony@students.uni-mainz.de)
# date:    November 2016
# Debug purpose: cuda-memcheck python main.py --GPU_global --CPU --outdir plots -b 4 -d 16
# python main.py --GPU_global --CPU --outdir plots -b 10 -d 5000 --use_given_edges
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import random as rnd
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gpu_hist

import os

FTYPE = np.float64

def mkdir(d, mode=0750, warn=True):
    """Simple wrapper around os.makedirs to create a directory but not raise an
    exception if the dir already exists

    Parameters
    ----------
    d : string
        Directory path
    mode : integer
        Permissions on created directory; see os.makedirs for details.
    warn : bool
        Whether to warn if directory already exists.

    """
    try:
        os.makedirs(d, mode=mode)
    except OSError as err:
        if err[0] == 17:
            if warn:
                print('Directory "%s" already exists' %d)
        else:
            raise err
    else:
        print('Created directory "%s"' %d)


def create_array(n_elements, n_dimensions):
    """Create an array with values between -360 and 360 (could be any other
    range too)"""
    # array = np.ndarray(shape(n_elements, n_dimensions), dtype=FTYPE)
    # for i in range(0, n_elements):
    #     for d in range(0, n_dimensions):
    #
    # return np.array(np.array(720*np.random.random(n_elements)-360,
    #         dtype=FTYPE) for i in range(0, n_dimensions))
    return np.array(720*np.random.random((n_elements, n_dimensions))-360,
            dtype=FTYPE)
    # return np.array(720*np.random.random((n_dimensions * n_elements))-360,
    #         dtype=FTYPE)


def create_weights(n_elements, n_dimensions):
    return np.random.random((n_dimensions, n_elements))


def create_edges(n_bins, n_dimensions):
    """Create some random edges given the number of bins for each dimension"""
    edges = []
    # Create some nice edges
    for d in range(0, n_dimensions):
        bin_width =720/n_bins
        edges_d =  np.arange(-360, 361, bin_width, dtype=self.FTYPE)
        edges.append(edges_d)
    return edges


# Currently only 1D and 2D
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
    ax.grid(b=True, which='major')
    ax.grid(b=True, which='minor', linestyle=':')
    print "Histogram:", name
    print np.sum(histogram)
    print histogram
    print "With edges:"
    print edges
    # print np.shape(edges)
    print np.shape(histogram)
    print len(np.shape(histogram))
    if(len(np.shape(histogram)) == 1):
        width = 60
        if edges is None:
            edges = np.arange(-360, 360, (720/no_of_bins))
            rects = ax.bar(edges, histogram, width)
            ax.set_xticks(edges)
            xtickNames = ax.set_xticklabels(edges)
        else:
            rects = ax.bar(edges[0][0:no_of_bins], histogram, width)
            ax.set_xticks(edges[0])
            xtickNames = ax.set_xticklabels(edges[0])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        fig.savefig(outdir+"/"+name)
    elif(len(np.shape(histogram)) == 2):
        X, Y = np.meshgrid(edges[0], edges[1])
        plt.pcolormesh(X, Y, histogram, cmap='rainbow')
        cbar = plt.colorbar(orientation='vertical')
        cbar.ax.tick_params(labelsize=9)
        ax.set_xticks(edges[0])
        ax.set_yticks(edges[1])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(9)
        # set the limits of the image
        plt.axis([X[0][0], X[0][len(X[0])-1], Y[0][0], Y[len(Y)-1][len(Y[len(Y)-1])-1]])
        fig.savefig(outdir+"/"+name)
    elif(len(np.shape(histogram)) == 3):
        print np.shape(edges)
        fig = plt.figure()
        for i in range(0, np.shape(histogram)[0]):
            subplot = len(edges[0])/2*100 + 20 + i + 1
            ax = fig.add_subplot(subplot)
            ax.grid(b=True, which='major')
            ax.grid(b=True, which='minor', linestyle=':')
            X, Y = np.meshgrid(edges[0], edges[1])
            plt.pcolormesh(X, Y, histogram[i], cmap='rainbow')
            cbar = plt.colorbar(orientation='vertical')
            cbar.ax.tick_params(labelsize=9)
            ax.set_xticks(edges[0])
            ax.set_yticks(edges[1])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(9)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(9)
            # set the limits of the image
            plt.axis([X[0][0], X[0][len(X[0])-1], Y[0][0], Y[len(Y)-1][len(Y[len(Y)-1])-1]])
        fig.savefig(outdir+"/"+name)

    else:
        print "Plots are only availale for 3 or less dimensions. Aborting"




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
    parser.add_argument('--CPU', action='store_true',
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
    parser.add_argument('--dimension', type=int, required=False, default=1,
            help=
            '''Define the number of dimensions for the input data and
            the histogram.''')
    parser.add_argument('-b', '--bins', type=int, required=False, default=256,
            help=
            '''Choose the number of bins for each dimension''')
    parser.add_argument('-w', '--weights', action='store_true',
            help=
            '''(Randomized) weights will be used on the histogram.''')
    parser.add_argument('--use_given_edges', action='store_true',
            help=
            '''Use calculated edges instead of calculating edges during
            histogramming.''')
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

    input_data = create_array(args.data, args.dimension)
    # print "input_data:\n", input_data
    # print np.shape(input_data)
    edges = None
    if args.use_given_edges:
        edges = create_edges(args.bins, args.dimension)
    print "Input_data: "
    print input_data
    print "----------------------"

    if args.full:
        print("Starting full histogramming")

        # First with double precision
        with gpu_hist.GPUHist(FTYPE=FTYPE, no_of_dimensions=args.dimension,
                no_of_bins=args.bins, edges=edges) as histogrammer:
            histogram_d_gpu_shared, edges_d_gpu_shared = histogrammer.get_hist(
                                                    input_data, shared=True)
            histogram_d_gpu_global, edges_d_gpu_global = histogrammer.get_hist(
                                                    input_data, shared=False)
        if edges is None:
            histogram_d_numpy, edges_d = np.histogramdd(input_data,
                    bins=args.bins, weights=weights)
        else:
            histogram_d_numpy, edges_d = np.histogramdd(input_data, bins=edges,
                    weights=weights)
        # Next with single precision
        FTYPE = np.float32
        with gpu_hist.GPUHist(FTYPE=FTYPE, no_of_dimensions=args.dimension,
                no_of_bins=args.bins, edges=edges) as histogrammer:
            histogram_s_gpu_shared, edges_s_gpu_shared = histogrammer.get_hist(
                                                    input_data, shared=True)
            histogram_s_gpu_global, edges_s_gpu_global = histogrammer.get_hist(
                                                    input_data, shared=False)
        if edges is None:
            histogram_s_numpy, edges_s = np.histogramdd(input_data,
                    bins=args.bins, weights=weights)
        else:
            histogram_s_numpy, edges_s = np.histogramdd(input_data, bins=edges,
                    weights=weights)
        if args.outdir != None:
            plot_histogram(histogram_d_gpu_shared, edges_d_gpu_shared,
                    args.outdir, "GPU shared memory, double", args.bins)
            plot_histogram(histogram_d_gpu_global, edges_d_gpu_global,
                    args.outdir, "GPU global memory, double", args.bins)
            plot_histogram(histogram_d_numpy, edges_d, args.outdir,
                    "CPU, double", args.bins)
            plot_histogram(histogram_s_gpu_shared, edges_s_gpu_shared,
                    args.outdir, "GPU shared memory, single", args.bins)
            plot_histogram(histogram_s_gpu_global, edges_s_gpu_global,
                    args.outdir, "GPU global memory, single", args.bins)
            plot_histogram(histogram_s_numpy, edges_s, args.outdir,
                    "CPU, single", args.bins)
        sys.exit()
    if args.GPU_both:
        print("Starting histogramming on GPU only")
        # if not args.all_precisions and args.single_precision then this is
        # single precision. Hence the missing "d" or "s" in the name.
        with gpu_hist.GPUHist(FTYPE=FTYPE, no_of_dimensions=args.dimension,
                no_of_bins=args.bins, edges=edges) as histogrammer:
            histogram_gpu_shared, edges_gpu_shared = histogrammer.get_hist(
                                                    input_data, shared=True)
            histogram_gpu_global, edges_gpu_global = histogrammer.get_hist(
                                                    input_data, shared=False)
        if args.all_precisions:
            FTYPE = np.float32
            with gpu_hist.GPUHist(FTYPE=FTYPE,
                    no_of_dimensions=args.dimension,
                    no_of_bins=args.bins, edges=edges) as histogrammer:
                histogram_s_gpu_shared, edges_s_gpu_shared = histogrammer.get_hist(input_data, shared=True)
                histogram_s_gpu_global, edges_s_gpu_global = histogrammer.get_hist(input_data, shared=False)
            plot_histogram(histogram_gpu_shared, edges_gpu_shared, args.outdir,
                    "GPU shared memory, double", args.bins)
            plot_histogram(histogram_gpu_global, edges_gpu_global, args.outdir,
                    "GPU global memory, double", args.bins)
            plot_histogram(histogram_s_gpu_shared, edges_s_gpu_shared,
                    args.outdir, "GPU shared memory, single", args.bins)
            plot_histogram(histogram_s_gpu_global, edges_s_gpu_global,
                    args.outdir, "GPU global memory, single", args.bins)
        else:
            name = ""
            if args.single_precision:
                name = "single"
            else:
                name = "double"
            plot_histogram(histogram_gpu_shared, edges_gpu_shared, args.outdir,
                    "GPU shared memory, " + name, args.bins)
            plot_histogram(histogram_gpu_global, edges_gpu_global, args.outdir,
                    "GPU global memory, " + name, args.bins)

    if args.GPU_shared and not args.GPU_both:
        print("Starting histogramming on GPU with shared memory")
        with gpu_hist.GPUHist(FTYPE=FTYPE, no_of_dimensions=args.dimension,
                no_of_bins=args.bins, edges=edges) as histogrammer:
            histogram_gpu_shared, edges_gpu_shared = histogrammer.get_hist(
                                                    input_data, shared=True)
        if args.all_precisions:
            FTYPE = np.float32
            with gpu_hist.GPUHist(FTYPE=FTYPE,
                    no_of_dimensions=args.dimension,
                    no_of_bins=args.bins, edges=edges) as histogrammer:
                histogram_s_gpu_shared, edges_s_gpu_shared = histogrammer.get_hist(input_data, shared=True)
            plot_histogram(histogram_gpu_shared, edges_gpu_shared, args.outdir,
                    "GPU shared memory, double", args.bins)
            plot_histogram(histogram_s_gpu_shared, edges_s_gpu_shared,
                    args.outdir, "GPU shared memory, single", args.bins)
        else:
            name = ""
            if args.single_precision:
                name = "single"
            else:
                name = "double"
            plot_histogram(histogram_gpu_shared, None, args.outdir,
                    "GPU shared memory, " + name, args.bins)

    if args.GPU_global and not args.GPU_both:
        print("Starting histogramming on GPU with global memory")
        with gpu_hist.GPUHist(FTYPE=FTYPE, no_of_dimensions=args.dimension,
                no_of_bins=args.bins, edges=edges) as histogrammer:
            histogram_gpu_global, edges_gpu_global = histogrammer.get_hist(
                                                    input_data, shared=False)
        if args.all_precisions:
            FTYPE = np.float32
            with gpu_hist.GPUHist(FTYPE=FTYPE,
                    no_of_dimensions=args.dimension,
                    no_of_bins=args.bins, edges=edges) as histogrammer:
                histogram_s_gpu_global, edges_s_gpu_global = histogrammer.get_hist(input_data, shared=False)
            plot_histogram(histogram_gpu_global, edges_gpu_global, args.outdir,
                    "GPU global memory, double", args.bins)
            plot_histogram(histogram_s_gpu_global, edges_s_gpu_global,
                    args.outdir, "GPU global memory, single", args.bins)
        else:
            name = ""
            if args.single_precision:
                name = "single"
            else:
                name = "double"
            plot_histogram(histogram_gpu_global, edges_gpu_global, args.outdir,
                    "GPU global memory, " + name, args.bins)

    if args.CPU:
        if edges is None:
            histogram_d_numpy, edges_d = np.histogramdd(input_data,
                    bins=args.bins, weights=weights)
        else:
            histogram_d_numpy, edges_d = np.histogramdd(input_data, bins=edges,
                    weights=weights)
        if args.all_precisions:
            FTYPE = np.float32
            if edges is None:
                histogram_s_numpy, edges_s = np.histogramdd(input_data,
                        bins=args.bins, weights=weights)
            else:
                histogram_s_numpy, edges_s = np.histogramdd(input_data,
                        bins=edges, weights=weights)
        if args.outdir != None:
            plot_histogram(histogram_d_numpy, edges_d, args.outdir,
                    "CPU, double", args.bins)
            if args.all_precisions:
                plot_histogram(histogram_s_numpy, edges_s, args.outdir,
                        "CPU, single", args.bins)
