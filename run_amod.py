#!/home/abcd/anaconda2/bin/python2.7

_description='''
This tool is used to generate pre-loadable(SW defined prototxt format <schema/DlaInterface.proto>) from caffe prototxt (<schema/caffe.proto>)
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

    parser.add_argument("--amod_dir",
            action="store",
            dest="amod_dir",
            required=True,
            help="AMOD root directory")

    parser.add_argument("--image_dir",
            action="store",
            dest="image_dir",
            required=True,
            help="The image directory, support .jpg only")
   
    parser.add_argument("--dst_dir",
            action="store",
            dest="dst_dir",
            required=True,
            help="The destination directory for output")

    parser.add_argument("--mean_file",
            action="store",
            dest="mean_file",
            required=True,
            help="The mean data file")
   
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

    print("hello")

    protofile = os.path.join(_args.amod_dir, 'models/bvlc_reference_rcnn_ilsvrc13/train_val_autogen.prototxt')
    caffe_proto = caffe_pb2.NetParameter()
    read_proto(caffe_proto, protofile)

    amod_exe = os.path.join(_args.amod_dir, './build/tools/caffe')
    #amod_exe = os.path.join(_args.amod_dir, '.build_release')
    weights = os.path.join(_args.amod_dir, 'models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel')
    #model = os.path.join(_args.amod_dir, 'models/bvlc_reference_rcnn_ilsvrc13/train_val_autogen.prototxt')
    model = os.path.join(_args.amod_dir, 'models/bvlc_reference_rcnn_ilsvrc13/train_val.prototxt')

    cmd = 'rm -rf ' + _args.dst_dir
    os.system(cmd)
    cmd = 'mkdir -p ' + _args.dst_dir
    os.system(cmd)

    # loop the input images
    for root, dirs, files in os.walk(_args.image_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.mat':
                full_filename = os.path.join(_args.image_dir, file)
                cmd = './pgm_loader/pgm_loader.py --input_file ' + full_filename + ' --dat_file temp.dat --mean_file ' + _args.mean_file
                logging.info(cmd)
                ret = os.system(cmd)
                if ret != 0:
                    logging.error('failed to convert %s to .dat file', full_filename)
                    return -1

                # Run AMOD
                cmd = amod_exe + ' test -iterations 1 -weights ' + weights + ' -model ' + model 
                logging.info(cmd)
                ret = os.system(cmd)
                if ret != 0:
                    logging.error('failed to run amod for %s', full_filename)
                    return -1

                # Save results
                dest_filename = os.path.join(_args.dst_dir, os.path.splitext(file)[0]+'.dat')
                cmd = 'mv fc_rcnn_fc_rcnn.dat ' +  dest_filename
                logging.info(cmd)
                ret = os.system(cmd)
            elif os.path.splitext(file)[1] == '.jpg':
    		print("Get JPG File")
                full_filename = os.path.join(_args.image_dir, file)
                cmd = './pgm_loader/pgm_loader.py --input_file ' + full_filename + ' --dat_file temp.dat --mean_file ' + _args.mean_file
                logging.info(cmd)
                ret = os.system(cmd)
                if ret != 0:
                    logging.error('failed to convert %s to .dat file', full_filename)
                    return -1

                # Run AMOD
                cmd = amod_exe + ' test -iterations 1 -weights ' + weights + ' -model ' + model 
                logging.info(cmd)
                ret = os.system(cmd)
                if ret != 0:
                    logging.error('failed to run amod for %s', full_filename)
                    return -1

                # Save results
                dest_filename = os.path.join(_args.dst_dir, os.path.splitext(file)[0]+'.dat')
                cmd = 'mv fc_rcnn_fc_rcnn.dat ' +  dest_filename
                logging.info(cmd)
                ret = os.system(cmd)
    return

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
