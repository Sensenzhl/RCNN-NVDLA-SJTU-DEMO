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
import threading


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

def merge_single_hist(dst, src):
    if dst.max_limit < src.max_limit:
        dst.max_limit = src.max_limit
    if dst.min_limit > src.min_limit:
        dst.min_limit = src.min_limit
    for i in range(0, src.num_marker):
        dst.hist_linear[i]   += src.hist_linear[i]
        dst.hist_exp[i]      += src.hist_exp[i]
        #logging.info("index:%d, linear hist:%d, exp hist:%d" % (i, src.hist_linear[i], src.hist_exp[i]))

    return

def write_proto( filename, proto):
    f = open(filename, "wb")
    f.write(proto_text.MessageToString(proto, False, False, False, False, '.15g'))
    f.close()
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



def is_lrn(type):
    return type == 'LRN'

def is_eltwise(type):
    return type == 'Eltwise'

def is_convolution(type):
    return type == 'Convolution' or type == 'InnerProduct' or type == 'Deconvolution'

def is_winograd(type, conv_param):
    if (type == 'InnderProduct'):
        return False
    else:
        kernel_h = 1
        kernel_w = 1
        if len(conv_param.kernel_size) > 1:
            kernel_h = conv_param.kernel_size[1]
            kernel_w = conv_param.kernel_size[0]
        elif len(conv_param.kernel_size) > 0:
            kernel_h = conv_param.kernel_size[0]
            kernel_w = conv_param.kernel_size[0]
        elif conv_param.HasField('kernel_w') and conv_param.HasField('kernel_h'):
            kernel_h = conv_param.kernel_h
            kernel_w = conv_param.kernel_w

        stride_x = 1
        stride_y = 1
        if len(conv_param.stride) > 1:
            stride_x = conv_param.stride[0]
            stride_y = conv_param.stride[1]
        elif len(conv_param.stride) > 0:
            stride_x = conv_param.stride[0]
            stride_y = conv_param.stride[0]
        elif conv_param.HasField('stride_w')  and conv_param.HasField('stride_h'):
            stride_x = conv_param.stride_w
            stride_y = conv_param.stride_h

        kernel_h_ext = (kernel_h + stride_y -1)/stride_y
        kernel_w_ext = (kernel_w + stride_x -1)/stride_y

        return kernel_h_ext == 3 and kernel_w_ext == 3

def is_need_dump(type):
    is_disable_dump = type == "Comparison" or \
                        type == "DataDistriWrapper" or \
                        type == "Dump" or \
                        type == "Loader" or \
                        type == "LUT" or \
                        type == "Silence"

    return not is_disable_dump

def merge_all_histogram(prefix, filecnt):
    hist = caffe_pb2.DataDistributeProto()
    is_initialized = False
    is_dstfile_valid = False
    dst_filename = prefix + ".txt"
    if not (os.path.isfile(dst_filename) and (_args.skip_existing == True)):
        if not os.path.isfile(prefix):
            for iter in range(filecnt):
                filename = prefix + str(iter)+".txt"
                if os.path.isfile(filename):
                    is_dstfile_valid = True
                    logging.info("load %d file:%s" % (iter, filename))
                    if is_initialized:
                        temp_hist = caffe_pb2.DataDistributeProto()
                        read_proto(temp_hist, filename)
                        merge_single_hist(hist, temp_hist)
                    else:
                        read_proto(hist, filename)
                        is_initialized = True;
                else:
                    logging.info("failed to open:%s" % (filename))

    if is_dstfile_valid == True:
        write_proto(dst_filename, hist)

    logging.info('%s merged', prefix)

    return

def formatting_name(name):
    new_name = name;
    new_name = new_name.replace('/', '_')
    new_name = new_name.replace('-', '_')

    return new_name

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(argc, argv):
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=_description)
    parser.add_argument('--version', action="version", version=__version__)

    parser.add_argument("--model",
            action="store",
            dest="model",
            required=True,
            help="The network definition in *.prototxt fomat")

    parser.add_argument("--file_num",
            action="store",
            dest="file_num",
            type=int,
            required=True,
            help="The number of files with the same prefix")

    parser.add_argument("--dir",
            action="store",
            dest="src_dir",
            required=True,
            help="The directory where your histogram are stored")

    parser.add_argument("--skip",
            action="store",
            dest="skip_existing",
            type=str2bool,
            default='False',
            required=False,
            help="Whether or not skip existing files")



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

    net = caffe_pb2.NetParameter()
    read_proto(net, _args.model)
    is_first_conv = True

    file_list = []
    for layer in net.layer:
        layer_name = formatting_name(layer.name)
        if is_convolution(layer.type):
            out_file = os.path.join(_args.src_dir, layer_name + '_datadistribute_')
            file_list.append(out_file)

            if is_winograd(layer.type, layer.convolution_param) and is_first_conv == False:
                logging.info('%s is winograd layer' % layer_name)
                # winograd convolution layer
                # weight histogram
                file = os.path.join(_args.src_dir, layer_name + '_param_0_datadistribute_')
                file_list.append(file)

                # weight pra histogram
                file = os.path.join(_args.src_dir, layer_name + '_weight_pra_datadistribute_')
                file_list.append(file)

                # feature pra histogram
                file = os.path.join(_args.src_dir, layer_name + '_feature_pra_datadistribute_')
                file_list.append(file)
                # feature cvt pra histogram
                file = os.path.join(_args.src_dir, layer_name + '_feature_cvt_pra_datadistribute_')
                file_list.append(file)
            else:
                logging.info('%s is dc conv layer' % layer_name)
                # weight histogram
                file = os.path.join(_args.src_dir, layer_name + '_param_0_datadistribute_')
                file_list.append(file)

            # bias histogram
            if layer.convolution_param.bias_term:
                file = os.path.join(_args.src_dir, layer_name + '_param_1_datadistribute_')
                file_list.append(file)
            is_first_conv = False
        elif is_lrn(layer.type):
            file = os.path.join(_args.src_dir, layer_name + '_lrn_intermidiate_datadistribute_')
            file_list.append(file)

            file = os.path.join(_args.src_dir, layer_name + '_datadistribute_')
            file_list.append(file)
        elif layer.type == 'SDPX':
            file = os.path.join(_args.src_dir, layer_name + '_datadistribute_')
            file_list.append(file)

            sdp_param = layer.sdp_x_param;
            if sdp_param.HasField('alu_op'):
                file = os.path.join(_args.src_dir, layer_name + '_param_0_datadistribute_')
                file_list.append(file)
            if sdp_param.HasField('mul_op'):
                file = os.path.join(_args.src_dir, layer_name + '_param_1_datadistribute_')
                file_list.append(file)
        elif is_eltwise(layer.type):
            file = os.path.join(_args.src_dir, layer_name + '_input_datadistribute_')
            file_list.append(file)

            if layer.eltwise_param.operation =='sum':
                file = os.path.join(_args.src_dir, layer_name + '_coef_datadistribute_')
                file_list.append(file)

            file = os.path.join(_args.src_dir, layer_name + '_datadistribute_')
            file_list.append(file)
        elif is_need_dump(layer.type):
            file = os.path.join(_args.src_dir, layer_name + '_datadistribute_')
            file_list.append(file)

    thread_pool = []
    # create threads
    for file in file_list:
        thread_pool.append(threading.Thread(target=merge_all_histogram, args=(file, _args.file_num)))

    # start threads
    for thread in thread_pool:
        thread.start()

    # join threads
    for thread in thread_pool:
        thread.join()

    return


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
