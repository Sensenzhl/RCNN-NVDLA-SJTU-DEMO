#!/home/scratch.yilinz_t19x/anaconda2/bin/python2.7

_description='''
This tool is used to  merge the distributed histograms to a single file
'''

import os
import inspect
import re
import sys
import argparse
import commands
import math
import logging
import copy
from pprint import pprint
from collections import OrderedDict
from google.protobuf import text_format as proto_text
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.stats import entropy


cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder += "/../build/python/caffe/"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

proto_dir = r"/home/utils/python-protobuf-2.6.1/lib/python2.7/site-packages/"
if proto_dir not in sys.path:
    sys.path.insert(0, proto_dir)

import caffe_pb2


class dict_of_dict(OrderedDict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return OrderedDict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

__version__ = '0.5'


def read_proto(proto, filename):
    # Read the existing address book.
    try:
        f = open(filename, "rb")
        proto_text.Merge(f.read(), proto)
        f.close()
    except IOError:
        logging.error(": Could not open file.  Creating a new one.")

    return



def normalize_distr(distr):
    summ = np.sum(distr)
    if summ != 0:
        distr = distr / summ

def entropy_calibration(bins):
    """entropy_calibration

    :param bins: histogram of abs(activations), np.array
    """
    bins = bins[:]
    bins[0] = bins[1]

    total_data = np.sum(bins)

    divergences = []
    arguments = []

    nbins = 128 # 128 = because we are quantizing to 128 values + sign
    stride = 1
    starting = 128
    stop = len(bins)

    new_density_counts = np.zeros(nbins, dtype=np.float64)

    for i in range(starting, stop + 1, stride):
        new_density_counts.fill(0)
        space = np.linspace(0, i, num=nbins + 1)
        digitized_space = np.digitize(range(i), space) - 1

        digitized_space[bins[:i] == 0] = -1

        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                new_density_counts[digitized] += bins[idx]

        counter = Counter(digitized_space)
        for key, val in counter.items():
            if key != -1:
                new_density_counts[key] = new_density_counts[key] / val

        new_density = np.zeros(i, dtype=np.float64)
        for idx, digitized in enumerate(digitized_space):
            if digitized != -1:
                new_density[idx] = new_density_counts[digitized]

        total_counts_new = np.sum(new_density) + np.sum(bins[i:])
        normalize_distr(new_density)

        reference_density = np.array(bins[:len(digitized_space)])
        reference_density[-1] += np.sum(bins[i:])

        total_counts_old = np.sum(reference_density)
        assert round(total_counts_new) == total_data
        assert round(total_counts_old) == total_data

        normalize_distr(reference_density)

        ent = entropy(reference_density, new_density)
        divergences.append(ent)
        arguments.append(i)

    divergences = np.array(divergences)
    last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
    result = last_argmin * stride + starting

    return result


def pre_process_hist(hist):
    updated_hist = []
    if hist.linear_mark[0] < 0:
        old_step = hist.linear_mark[1] - hist.linear_mark[0]
        fabs_max = max(math.fabs(hist.max_limit), math.fabs(hist.min_limit))
        total_entry = fabs_max/old_step
        temp_hist   = np.zeros(int(math.ceil(total_entry)))

        for iter in range(hist.num_marker):
            sample  = math.fabs(hist.linear_mark[iter] - old_step/2)
            idx     = int(math.ceil(sample/old_step))
            temp_hist[idx] += hist.hist_linear[iter]

        updated_hist = temp_hist
    else:
        updated_hist = hist
        

    return updated_hist


def main(argc, argv):
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=_description)
    parser.add_argument('--version', action="version", version=__version__)

    parser.add_argument("--file",
            action="store",
            dest="src_file",
            required=True,
            help="The input histogram file")

    parser.add_argument("--log",
            action="store",
            dest="log_file",
            required=False,
            default='',
            help="Debug message will be output to this file")

    global _args

    FORMAT = '%(asctime)-15s %(message)s'
    _args = parser.parse_args()
    if _args.log_file:
        logging.basicConfig(format=FORMAT, filename=_args.log_file, filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(format=FORMAT, level=logging.INFO)

    hist = caffe_pb2.DataDistributeProto()
    if os.path.isfile(_args.src_file):
        read_proto(hist, _args.src_file)
    else:
        logging.error('cant open: %s' % _args.src_file)

    bins            = pre_process_hist(hist)
    opt_hist_index  = entropy_calibration(bins)

    print(opt_hist_index * (hist.linear_mark[1] - hist.linear_mark[0]))

    return


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
