#!/home/scratch.yilinz_t19x/anaconda2/bin/python2.7

_description='''
This tool is used to viisualize the histogram collected from real network
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


#################### Global Variables ######################

def merge_distr(dst, src):
    if dst.max_limit < src.max_limit:
        dst.max_limit = src.max_limit
    if dst.min_limit > src.min_limit:
        dst.min_limit = src.min_limit
    for i in range(0, src.num_marker):
        dst.hist_linear[i]   += src.hist_linear[i]
        dst.hist_exp[i]      += src.hist_exp[i]
        #logging.info("index:%d, linear hist:%d, exp hist:%d" % (i, src.hist_linear[i], src.hist_exp[i]))

    return

def read_proto(proto, filename):
    # Read the existing address book.
    try:
        f = open(filename, "rb")
        proto_text.Merge(f.read(), proto)
        f.close()
    except IOError:
        logging.error(": Could not open file.  Creating a new one.")

    return

def main(argc, argv):
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=_description)
    parser.add_argument('--version', action="version", version=__version__)

    parser.add_argument("--prefix",
            action="store",
            dest="src_file",
            required=True,
            help="The prefix/filename of histogram files")

    parser.add_argument("--output",
            action="store",
            dest="dst_file",
            required=True,
            help="The output filename of generated histogram diagram")

    parser.add_argument("--mode",
            action="store",
            dest="mode",
            required=False,
            default=0,
            help="Select which histogram should be visualzied, 0 for linear, 1 for exponent")


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

    distri = caffe_pb2.DataDistributeProto()
    is_initialized = False;

    if os.path.isfile(_args.src_file):
        logging.info("load single file:%s" % (_args.src_file))
        read_proto(distri, _args.src_file)
    else:
        file_index = 0
        while os.path.isfile(_args.src_file+str(file_index)+".txt"):
            filename = _args.src_file + str(file_index)+".txt"
            logging.info("load %d file:%s" % (file_index, filename))
            if is_initialized:
                temp_distri = caffe_pb2.DataDistributeProto()
                read_proto(temp_distri, filename)
                merge_distr(distri, temp_distri)
            else:
                read_proto(distri, filename)
                is_initialized = True;
            file_index += 1

    start_index = 0
    end_index   = len(distri.hist_linear)
    marker  = []
    hist    = []
    if (_args.mode == 0):
        samples = sum(distri.hist_linear)
        temp_hist = [ float(x)/float(samples) for x in distri.hist_linear]
        for i in range(end_index):
            if temp_hist[i] != 0:
                start_index = i
                break
        for i in range(end_index-1,-1,-1):
            if temp_hist[i] != 0:
                end_index = i
                break
        marker  = distri.linear_mark[start_index:end_index]
        hist    = temp_hist[start_index:end_index]
    else:
        samples = sum(distri.hist_exp)
        temp_hist = [ float(x)/float(samples) for x in distri.hist_exp]
        for i in range(end_index):
            if temp_hist[i] != 0:
                start_index = i
                break
        for i in range(end_index-1,-1,-1):
            if temp_hist[i] != 0:
                end_index = i
                break
        marker  = distri.exp_mark[start_index:end_index]
        hist    = temp_hist[start_index:end_index]
    
    logging.info('visualized samples occupied:%f of total samples', sum(hist))
    plt.plot(marker, hist)
    plt.savefig(_args.dst_file)
   

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
