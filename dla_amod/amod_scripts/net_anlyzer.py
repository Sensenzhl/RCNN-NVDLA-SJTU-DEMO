#!/home/scratch.yilinz_t19x/anaconda2/bin/python2.7

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
from pprint import pprint
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

proto_dir = r"/home/utils/python-protobuf-2.6.1/lib/python2.7/site-packages/"
if proto_dir not in sys.path:
    sys.path.insert(0, proto_dir)

from google.protobuf import text_format as proto_text
import caffe_pb2
import caffe


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

def read_txt_proto(proto, filename):
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

def read_bin_proto(proto, filename):
    """
    Read the existing address book.

    Returns:
        The proto structure
    """
    try:
        f = open(filename, "rb")
        proto.ParseFromString(f.read())
        f.close()
    except IOError:
        logging.error(": Could not open file.  Creating a new one.")

    return proto


def is_convolution(type):
    return type == 'Convolution' or type == 'InnerProduct' or type == 'Deconvolution'

TOTAL_COMPRESSED_WT_SIZE        = 0
TOTAL_COMPRESSED_MASK_SIZE      = 0
TOTAL_WT_SIZE                   = 0

def analyze_weight_zeros(net):
    # read the weights
#    caffe_net   = caffe.Net(str(net.prototxt_file), caffe.TEST)
#    logging.info('finished setup:%s\n', net.prototxt_file)
#    caffe_net.copy_from(str(net.weight_file))
#    logging.info('finished load weigths\n')
    model       = caffe_pb2.NetParameter()

    global TOTAL_COMPRESSED_MASK_SIZE
    global TOTAL_COMPRESSED_WT_SIZE
    global TOTAL_WT_SIZE
    logging.info('load weight:%s\n', net.weight_file)
    model = read_bin_proto(model, str(net.weight_file))

    logging.info('layer name:%s', model.name)

    layer_name_list             = list()
    layer_compress_mask_list    = list()
    layer_compress_nz_list      = list()
    idx                         = list()

    logging.info('weight layer:%d, layers:%d, input_shape:%d, input_dim:%d\n', len(model.layer), len(model.layers), len(model.input_shape), len(model.input_dim))
    net_total_size      = 0;
    net_compressed_size = 0;
    layer_index         = 0
    if len(model.layer) > 0:
        for layer in model.layer:
            if is_convolution(layer.type):
                wt_blob = layer.blobs[0]
                assert len(layer.blobs) > 0
                if len(layer.blobs[0].data) > 0:
                    wt = np.array(layer.blobs[0].data)
                else:
                    assert len(layer.blobs[0].double_data) > 0
                    wt = np.array(layer.blobs[0].double_data)

                if wt_blob.HasField('shape'):
                    assert len(wt_blob.shape.dim) == 4
                    num     = wt_blob.shape.dim[0]
                    channel = wt_blob.shape.dim[1]
                    height  = wt_blob.shape.dim[2]
                    width   = wt_blob.shape.dim[3]
                else:
                    num     = wt_blob.num
                    channel = wt_blob.channels
                    height  = wt_blob.height
                    width   = wt_blob.width
                wt_reshaped = wt.reshape(num, \
                                         channel*height*width)
                logging.info('layer:%s, weight shape:%s', layer.name, wt.shape)

                # convert weight to int8
                for i in range(0, num):
                    scale = 128.0/np.amax(wt_reshaped[i,:])
                    wt_reshaped[i,:] = np.around(wt_reshaped[i,:]*scale)

                # calculate the compressed weight size
                total_size  = num*channel*height*width
                num_nz      = np.count_nonzero(wt_reshaped)
                mask_sz     = total_size/8
                compressed_wt_size = mask_sz + num_nz

                TOTAL_WT_SIZE               += total_size
                net_total_size              += total_size
                if (compressed_wt_size < total_size):
                    TOTAL_COMPRESSED_MASK_SIZE  += mask_sz
                    TOTAL_COMPRESSED_WT_SIZE    += num_nz

                    net_compressed_size         += compressed_wt_size
                else:
                    # compress doesn't save bandwidth, let's use normal mode
                    TOTAL_COMPRESSED_MASK_SIZE  += 0 
                    TOTAL_COMPRESSED_WT_SIZE    += total_size
                    net_compressed_size         += total_size 

                layer_name_list.append(layer.name + '(' + layer.type + ')')
                layer_compress_mask_list.append(float(mask_sz)/total_size)
                layer_compress_nz_list.append(float(num_nz)/total_size)
                layer_index += 1
                idx.append(layer_index)
    elif len(model.layers) > 0:
        for layer in model.layers:
            if layer.type == caffe_pb2.V1LayerParameter.CONVOLUTION or \
                layer.type == caffe_pb2.V1LayerParameter.INNER_PRODUCT:
                logging.info('parse %s\n', layer.name)
                assert len(layer.blobs) > 0
                wt_blob = layer.blobs[0]
                if len(layer.blobs[0].data) > 0:
                    wt = np.array(layer.blobs[0].data)
                else:
                    assert len(layer.blobs[0].double_data) > 0
                    wt = np.array(layer.blobs[0].double_data)
                if wt_blob.HasField('shape'):
                    assert len(wt_blob.shape.dim) == 4
                    num     = wt_blob.shape.dim[0]
                    channel = wt_blob.shape.dim[1]
                    height  = wt_blob.shape.dim[2]
                    width   = wt_blob.shape.dim[3]
                    if num*channel == 1:
                        num = height
                        height = 1
                    logging.info('shape:%dx%dx%dx%d', num, channel, height, width)
                else:
                    num     = wt_blob.num
                    channel = wt_blob.channels
                    height  = wt_blob.height
                    width = wt_blob.width
                    if num*channel == 1:
                        num = height
                        height = 1

                    logging.info('shape:%dx%dx%dx%d', num, channel, height, width)
                wt_reshaped = wt.reshape(num, \
                                         channel*height*width)
                logging.info('layer:%s, weight shape:%s', layer.name, wt_reshaped.shape)

                # convert weight to int8
                for i in range(0, num):
                    scale = 128.0/np.amax(wt_reshaped[i,:])
                    wt_reshaped[i,:] = np.around(wt_reshaped[i,:]*scale)

                # calculate the compressed weight size
                total_size  = num*channel*height*width 
                num_nz      = np.count_nonzero(wt_reshaped)
                mask_sz     = total_size/8
                compressed_wt_size = mask_sz + num_nz
                logging.info('num_nz:%d, total_size:%d, zero ratio:%f\n', num_nz, total_size, float(total_size-num_nz)/total_size)


                TOTAL_WT_SIZE               += total_size
                net_total_size              += total_size
                if (compressed_wt_size < total_size):
                    logging.info('use weight compress for layer:%s', layer.name)
                    TOTAL_COMPRESSED_MASK_SIZE  += mask_sz
                    TOTAL_COMPRESSED_WT_SIZE    += num_nz

                    net_compressed_size         += compressed_wt_size
                else:
                    logging.info('deprecate weight compress for layer:%s', layer.name)
                    # compress doesn't save bandwidth, let's use normal mode
                    TOTAL_COMPRESSED_MASK_SIZE  += 0 
                    TOTAL_COMPRESSED_WT_SIZE    += total_size
                    net_compressed_size         += total_size 

                #layer_name_list.append(layer.name + '(' + caffe_pb2.V1LayerParameter.LayerType.values_by_number[layer.type].name + ')')
                layer_name_list.append(layer.name + '(' + str(layer.type) + ')')
                layer_compress_mask_list.append(float(mask_sz)/total_size)
                layer_compress_nz_list.append(float(num_nz)/total_size)
                layer_index += 1
                idx.append(layer_index)


    # plot
    p1 = plt.bar(idx, layer_compress_mask_list, color='red', edgecolor='black', hatch="/")
    p2 = plt.bar(idx, layer_compress_nz_list, bottom=layer_compress_mask_list, \
                 color='green', edgecolor='black', hatch="//")
    plt.ylabel('ratio')
    plt.title(model.name + ' weight compress ratio')
    plt.xticks(idx, layer_name_list)
    plt.yticks(np.arange(0, 1, 0.1))
    #plt.legend((p1[0], p2[0]), ('mask', 'non-zeros'))
    plt.savefig(os.path.join(_args.dst_dir, model.name+'.jpg'))

    plt.gcf().clear()

    return float(net_compressed_size)/float(net_total_size)
    

def main(argc, argv):
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=_description)
    parser.add_argument('--version', action="version", version=__version__)

    parser.add_argument("--nets",
            action="store",
            dest="nets",
            required=True,
            help="The network definition in *.prototxt format")
    parser.add_argument("--dst_dir",
            action="store",
            dest="dst_dir",
            required=True,
            help="The directory to save the plotted figures")

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

    # Read the network yellow book
    nets = caffe_pb2.Nets()
    read_txt_proto(nets, _args.nets)
    caffe.set_mode_cpu()

    net_names                   = list()
    net_weight_compress_ratio   = list()
    for net in nets.net:
        net_weight_compress_ratio.append(analyze_weight_zeros(net))
        net_names.append(net.name)




    # Weight statistics for entire network lists
    idx = range(0, len(nets.net))
    p1 = plt.bar(idx, net_weight_compress_ratio, color='green')
    plt.ylabel('ratio')
    plt.title('SUMMAYR: weight compress ratio')
    plt.xticks(idx, net_names)
    plt.yticks(np.arange(0, 1, 0.1))
    plt.savefig(os.path.join(_args.dst_dir, 'summary_weight_compress.jpg'))
    plt.gcf().clear()



    return

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
