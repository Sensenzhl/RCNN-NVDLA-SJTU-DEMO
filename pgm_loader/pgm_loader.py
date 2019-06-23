#!/usr/bin/python2.7

_description='''
This tool is used to analyze the network parameters
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
import cv2
from pprint import pprint
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import misc
import scipy.io as scio  
import imageio
import matplotlib.image as mpimg

proto_dir = r"/home/abcd/anaconda2/lib/python2.7/site-packages"
if proto_dir not in sys.path:
    sys.path.insert(0, proto_dir)

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

def write_proto(proto, filename):
    """
    Write the proto to file 

    Returns:
        N/A
    """
    f = open(filename, "wb")
    f.write(proto_text.MessageToString(proto, False, False, False, False, '.15g'))
    f.close()

    return

def read_image(filename):
    """
    Read the pgm image

    Returns:
       The structure contains image data 
    """

    img = []

    if os.path.splitext(filename)[1] == '.pgm':
        pgmf = open(filename, 'rb')

        assert pgmf.readline() == 'P5\n'
        (width, height, depth) = [int(i) for i in pgmf.readline().split()]
        assert depth <= 255
        logging.info('image dimension: WxH=%dx%d', width, height)

        for iter in range(height*width):
            img.append(ord(pgmf.read(1)))

        pgmf.close()
    elif os.path.splitext(filename)[1] == '.mat':
        data = scio.loadmat(filename)  
        if _args.mean_file != '':
            logging.info('read mean file:%s\n', _args.mean_file)
            mean = scio.loadmat(_args.mean_file)            
            assert mean['image_mean'].shape == data['window2'].shape

            data['window2'] -= mean['image_mean']

        (width, height, channel ) = data['window2'].shape

        # Convert the channel first to width first order
        (width, height, channel) = data['window2'].shape
        image = []
        for c in range(0, channel):
            sliced = data['window2'][:,:,c]
            image.append([item for sublist in sliced.tolist() for item in sublist])
        for c in range(0, channel):
            img += image[c]
    else:
        #print("jpg_file:"+filename)
        #data = misc.imread(filename)
        #data  = np.asarray(Image.open(filename))
        data = cv2.imread(filename)
        #data = np.transpose(data,(2,1,0))
        #data2 = mpimg.imread(filename)
        #data3 = io.imread(filename)
        #data = cv2.imread(filename)

        data = data[:,:,[2,1,0]]

        #print("Opencv")
        #print(data[0,0:10,0])
        #print(data[0,0:10,1])
        #print(data[0,0:10,2])  

        #print("Mpimg")
        #print(data2[0,0:10,0])
        #print(data2[0,0:10,1])
        #print(data2[0,0:10,2])           

        #print("IO-imread")
        #print(data3[0,0:10,0])
        #print(data3[0,0:10,1])
        #print(data3[0,0:10,2])      

        data_f = data.astype(float)
        if _args.mean_file != '':
            logging.info('read mean file:%s\n', _args.mean_file)
            mean = scio.loadmat(_args.mean_file)
            assert mean['image_mean'].shape == data.shape

            if False:
#            if True:
                mean_i = mean['image_mean'].astype(int)
                mean_f = mean_i.astype(float)
                data_f -= mean_f
            else:
                data_f -= mean['image_mean']

        width   = data.shape[1]
        height  = data.shape[0]
        channel = data.shape[2]
        data_f = data_f.transpose((1,0,2))

        image = []
        for c in range(0, channel):
            sliced = data_f[:,:,c]
            image.append([item for sublist in sliced.tolist() for item in sublist])
        for c in range(0, channel):
            img += image[c]

        assert width    == 227
        assert height   == 227

        assert len(img) == width*height*channel

    return (width, height, channel, img)


def main(argc, argv):
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=_description)
    parser.add_argument('--version', action="version", version=__version__)

    parser.add_argument("--input_file",
            action="store",
            dest="input_file",
            required=True,
            help="The input pgm filename")

    parser.add_argument("--mean_file",
            action="store",
            dest="mean_file",
            required=False,
            default='',
            help="The per-element mean data file")

    parser.add_argument("--dat_file",
            action="store",
            dest="dat_file",
            required=True,
            help="The output .dat filename")

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

    (width, height, channel, img) = read_image(_args.input_file)

    blobs   = caffe_pb2.BlobProtoVector()
    blob    = blobs.blobs.add()

    # Generate a blob object
    shape       = caffe_pb2.BlobShape()
    shape.dim[:] = [1, channel, height, width]
    blob.shape.CopyFrom(shape)
    for item in img:
        blob.double_data.append(float(item))

    write_proto(blobs, _args.dat_file)

    return 0

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
