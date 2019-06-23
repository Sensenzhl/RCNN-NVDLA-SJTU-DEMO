#!/home/scratch.yilinz_t19x/anaconda2/bin/python2.7

_description='''
This tool is used to  shrink the distributed histograms to a single file
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
                        type == "LUT"

    return not is_disable_dump

def shrink_histogram(src_filename, dst_dir, entry_cnt):
    src_hist    = caffe_pb2.DataDistributeProto()
    dst_hist    = caffe_pb2.DataDistributeProto()
    dst_filename = os.path.join(dst_dir, os.path.basename(src_filename + ".txt"))

    logging.info('read %s', src_filename)
    read_proto(src_hist, src_filename + '.txt');

    dst_hist.num_marker = entry_cnt
    dst_hist.scale      = src_hist.scale
    dst_hist.max_limit  = src_hist.max_limit
    dst_hist.min_limit  = src_hist.min_limit
    dst_hist.avg        = src_hist.avg
    dst_hist.max_sum_weight = src_hist.max_sum_weight

    inner_step = (src_hist.num_marker-1)/(entry_cnt-1)
    assert inner_step == int(inner_step)

    start_marker = src_hist.linear_mark[0]
    marker_step = (src_hist.linear_mark[1] - src_hist.linear_mark[0])*inner_step
    dst_hist.linear_mark.append(src_hist.linear_mark[0])
    dst_hist.hist_linear.append(src_hist.hist_linear[0])
    for i in range(1, entry_cnt-1):
        dst_hist.linear_mark.append(i*marker_step+start_marker)
        samples = 0
        for j in range(i*inner_step, (i-1)*inner_step, -1):
            samples = samples + src_hist.hist_linear[j]
        dst_hist.hist_linear.append(samples)
    
    write_proto(dst_filename, dst_hist)

    logging.info('%s shrinked to %s', src_filename, dst_filename)

    return

def formatting_name(name):
    new_name = name;
    new_name.replace('/', '_')
    new_name.replace('-', '_')

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

    parser.add_argument("--i_dir",
            action="store",
            dest="src_dir",
            required=True,
            help="The directory where your histogram are stored")

    parser.add_argument("--o_dir",
            action="store",
            dest="dst_dir",
            required=True,
            help="The directory where your histogram are stored")

    parser.add_argument("--entry_num",
            action="store",
            dest="entry_num",
            type=int,
            required=True,
            help="The directory where your histogram are stored")


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

    # clear the dst dir
    cmd = 'rm -rf ' + _args.dst_dir
    os.system(cmd)
    cmd = 'mkdir -p ' + _args.dst_dir
    os.system(cmd)

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
            file = os.path.join(_args.src_dir, layer_name + '_param_1_datadistribute_')
            file_list.append(file)
            is_first_conv = False
        elif is_lrn(layer.type):
            file = os.path.join(_args.src_dir, layer_name + '_lrn_intermidiate_datadistribute_')
            file_list.append(file)

            file = os.path.join(_args.src_dir, layer_name + '_datadistribute_')
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
        thread_pool.append(threading.Thread(target=shrink_histogram, args=(file, _args.dst_dir, _args.entry_num)))

    # start threads
    for thread in thread_pool:
        thread.start()

    # join threads
    for thread in thread_pool:
        thread.join()

    return


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
