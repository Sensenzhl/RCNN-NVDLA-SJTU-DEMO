#!/usr/bin/python3

_description='''
This tool is used to generate pre-loadable(SW defined prototxt format <schema/DlaInterface.proto>) from caffe prototxt (<schema/caffe.proto>)
'''

import os
import inspect
import re
import sys
import argparse
#import commands
import math
import logging
import copy
from pprint import pprint
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import ctypes
from PIL import Image
from google.protobuf import text_format as proto_text
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

def read_proto(proto, filename):
    """
    Read the existing address book.

    Returns:
        The proto structure
    """
    try:
        f = open(filename, "rb")
        proto_text.Merge(f.read(), proto)
        f.close()
    except IOError:
        logging.error(": Could not open file.  Creating a new one.")

    return

def write_proto( filename, proto):
    f = open(filename, "wb")
    f.write(proto_text.MessageToString(proto, False, False, False, False, '.15g'))
    f.close()
    return



def main(argc, argv):
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=_description)
    parser.add_argument('--version', action="version", version=__version__)

    parser.add_argument("--image_dir",
            action="store",
            dest="image_dir",
            required=True,
            help="The image directory, support .jpg only")
  
    parser.add_argument("--log",
            action="store",
            dest="log_file",
            required=False,
            default='',
            help="Debug message will be output to this file")

    parser.add_argument("--mean_file",
            action="store",
            dest="mean_file",
            required=True,
            help="The mean data file")

    parser.add_argument("--dst_dir",
            action="store",
            dest="dst_dir",
            required=False,
            default='./output',
            help="Output Folder")

    global _args

    FORMAT = '%(asctime)-15s %(message)s'
    _args = parser.parse_args()
    if _args.log_file:
        logging.basicConfig(format=FORMAT, filename=_args.log_file, filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(format=FORMAT, level=logging.INFO)

    print("Ready to Go!")
    # loop the input images
    for root, dirs, files in os.walk(_args.image_dir): 
        #for my_dir in dirs:
        cmd = ('./run_amod.py --amod_dir=./dla_amod --image_dir=' + os.path.join(_args.image_dir) + 
               ' --dst_dir ' + _args.dst_dir + ' --mean_file=' + _args.mean_file)
        os.system(cmd)
    return

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
