#!/home/scratch.yilinz_t19x/anaconda2/bin/python2.7

_description='''
This tool is used to get per-layer averaging difference between FP32 golden and DLA modelized 
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

def display_data_average(_data_base):
    for layer_name in _data_base.keys():
        for blob_type in _data_base[layer_name].keys():
            for blob_name in _data_base[layer_name][blob_type].keys():
                logging.info("%s %s %s average:%f" % (layer_name, blob_type, blob_name,
                                                      _data_base[layer_name][blob_type][blob_name]['data_average']))

def calc_data_average (_data_base, directory):
    grep_cmd = "find " + directory + " -name test.stderr | xargs grep \"data:\" |  grep -v \"Memory\""
    source_lines = commands.getoutput(grep_cmd)
    source_lines = source_lines.split("\n")

    line_num = 0
    _data_base.clear()

    for line in source_lines:

#     [-+]? # optional sign
#     (?:
#         (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
#         |
#         (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
#     )
#     # followed by optional exponent part if desired
#     (?: [Ee] [+-]? \d+ ) ?
#        m = re.search(r".*\[Forward\] Layer (?P<layer_name>\w+), (?P<blob_type>\w+) blob (?P<blob_name>\w+) data: (?P<data>\d+\.\d+) dimension", line)
        m = re.search(r".*\[Forward\] Layer (?P<layer_name>[\/\w-]+), (?P<blob_type>[\/\w-]+) blob (?P<blob_name>[\/\w-]+) data: (?P<data>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?),? dimension", line)
        if m:
            layer_name  = m.group('layer_name')
            blob_type   = m.group('blob_type')
            blob_name   = m.group('blob_name')
            data        = m.group('data')

            layer_name = layer_name.replace(r'/', '_')
            layer_name = layer_name.replace('-', '_')
            blob_name = blob_name.replace(r'/', '_')
            blob_name = blob_name.replace('-', '_')

            if layer_name in _data_base.keys():
                ## layer name is existed
                if blob_type in _data_base[layer_name].keys():
                    ## blob type is existed
                    if blob_name in _data_base[layer_name][blob_type].keys():
                        ## blob name is existed
                        _data_base[layer_name][blob_type][blob_name]['data_sum'] += float(data)
                        _data_base[layer_name][blob_type][blob_name]['data_count'] += 1
                    else:
                        ## blob name is not existed, got a new blob name
                        _data_base[layer_name][blob_type][blob_name]['data_sum'] = float(data)
                        _data_base[layer_name][blob_type][blob_name]['data_count'] = 1
                else:
                    ## blob type is not existed, got a new blob type
                    _data_base[layer_name][blob_type][blob_name]['data_sum'] = float(data)
                    _data_base[layer_name][blob_type][blob_name]['data_count'] = 1
            else:
                ## layer_name is not existed, got a new layer name
                _data_base[layer_name][blob_type][blob_name]['data_sum'] = float(data)
                _data_base[layer_name][blob_type][blob_name]['data_count'] = 1
        else:
            logging.info("Parse failed in line %d: %s" % (line_num, line))
        line_num += 1

    logging.debug('parsed {line:d}'.format(line=line_num))
    item_count = 0
    ## Calculate the average
    for layer_name in _data_base.keys():
        for blob_type in _data_base[layer_name].keys():
            for blob_name in _data_base[layer_name][blob_type].keys():
                _data_base[layer_name][blob_type][blob_name]['data_average'] =\
                        _data_base[layer_name][blob_type][blob_name]['data_sum']/_data_base[layer_name][blob_type][blob_name]['data_count']


    #display_data_average(_data_base)
    return

def get_delta(dict_golden_data, dict_cur_data, model_scale, model_offset, model, model_scale2, model_offset2):
    global _args
    blob_type = 'top'
    for layer_name in dict_golden_data.keys():
        for blob_name in dict_golden_data[layer_name][blob_type].keys():
            if (layer_name in dict_cur_data.keys()) and \
                    (blob_name in dict_cur_data[layer_name][blob_type].keys()) and \
                    is_layer_contained(model, layer_name):
                delta  = math.fabs(dict_golden_data[layer_name][blob_type][blob_name]['data_average'] -\
                             dict_cur_data[layer_name][blob_type][blob_name]['data_average'])
                logging.info('layer: {layer:30s}, blob: {blob:26s}, data delta: {delta:10f}, data1:{golden:10f}, data2:{cur:10f}, scale1: {scale:10f}, offset1: {offset:10f}, scale2: {scale2:10f}'.format(\
                                layer=layer_name, blob=blob_name, delta=delta, scale=model_scale[layer_name], scale2=model_scale2[layer_name], \
                                offset=model_offset[layer_name],\
                                golden=dict_golden_data[layer_name][blob_type][blob_name]['data_average'],\
                                cur=dict_cur_data[layer_name][blob_type][blob_name]['data_average']))
            else:
                logging.debug('layer:{layer:s}, blob:{blob:s} doesnt found in reference data array'.format(
                    layer=layer_name, blob=blob_name) +
                              ' layer_sts:{layer_sts:d}, blob_sts:{blob_sts:d}, contained:{contained:d}'.format(
                              layer_sts = layer_name in dict_cur_data.keys(),
                              blob_sts = blob_name in dict_cur_data[layer_name][blob_type].keys(),
                              contained= is_layer_contained(model, layer_name)))

    return

def is_convolution_layer(layer_type):
    assert layer_type != "Deconvolution"
    return layer_type == "Convolution"

def is_inner_product_layer(layer_type):
    return layer_type == "InnerProduct"

def is_relu_layer(layer_type):
    return layer_type == "ReLU"

def is_sigmoid_layer(layer_type):
    return layer_type == 'Sigmoid'

def is_lut_activation_layer(layer_type):
    return layer_type == "Sigmoid" or layer_type == "TanH"

def is_software_layer(layer_type):
    return layer_type == "Softmax" or layer_type == 'Accuracy' or layer_type == 'SoftmaxWithLoss'

def is_lrn_layer(layer_type):
    return layer_type == "NVLRN"

def is_lut_layer(layer_type):
    return is_lrn_layer(layer_type) or is_lut_activation_layer(layer_type)

def is_convertor_layer( layer_type):
    return layer_type == "Convertor"

# Layers are added by calibration, it's not existed in original prototxt
# thus we don't have histogram for those layers
def has_histogram( layer_type):
    return layer_type != "Convertor" and layer_type != "Pooling"


def get_conv_weight_scale( conv_param, is_inner_product ):
    scale = 1.0
    if is_inner_product or conv_param.engine != caffe_pb2.ConvolutionParameter.WINOGRAD:
        if conv_param.HasField("weight_convert"):
            if conv_param.weight_convert.scale_method == caffe_pb2.GLOBAL_SCALING:
                to_coef = conv_param.weight_convert.to_coef
                scale = to_coef.scale*to_coef.post_scale
    else:
        if conv_param.HasField("pra_weight_convert"):
            if conv_param.pra_weight_convert.scale_method == caffe_pb2.GLOBAL_SCALING:
                to_coef = conv_param.pra_weight_convert.to_coef
                scale = to_coef.scale*to_coef.post_scale

    return scale

def convertor_scaling(convert_param ):
    scale = 1.0

    if convert_param.scale_method == caffe_pb2.GLOBAL_SCALING:
        to_coef = convert_param.to_coef
        scale = float(to_coef.scale)*float(to_coef.post_scale)/pow(2.0, to_coef.shifter)

    return scale

def convertor_offset(convert_param, base_scale, base_offset ):
    offset = 0.0


    if convert_param.scale_method == caffe_pb2.GLOBAL_SCALING:
        to_coef = convert_param.to_coef
        offset = to_coef.offset/base_scale + base_offset

    return offset

def get_conv_internal_scale( conv_param, is_inner_product ):
    scale = 1.0

    # PRA feature scale
    if is_inner_product == False and conv_param.engine == caffe_pb2.ConvolutionParameter.WINOGRAD \
        and conv_param.pra_feature_truncat.scale_method == caffe_pb2.GLOBAL_SCALING:
        if conv_param.HasField("pra_weight_convert"):
            to_coef = conv_param.pra_feature_truncat.to_coef
            scale = to_coef.scale*to_coef.post_scale/pow(2.0, to_coef.shifter)

    # output shifter
    if conv_param.HasField("output_truncat") and conv_param.output_truncat.scale_method == caffe_pb2.GLOBAL_SCALING:
        to_coef = conv_param.output_truncat.to_coef
        scale *= to_coef.scale*to_coef.post_scale/pow(2.0, to_coef.shifter)

    return scale

def update_scale( model_scale, model_offset, model ):
    bottom2layer    = dict_of_dict()
    top2layer       = dict_of_dict()
    base_scale      = 1.0
    base_offset     = 0

    for layer in model.layer:
        layer_name = layer.name.replace(r'/', '_')
        layer_name = layer_name.replace('-', '_')

        if len(layer.bottom) > 0:
            bottom_blob_name = layer.bottom[0].replace(r'/', '_')
            bottom_blob_name = bottom_blob_name.replace('-', '_')
            if bottom_blob_name in top2layer.keys():

                base_scale = model_scale[top2layer[bottom_blob_name]]
                base_offset= model_offset[top2layer[bottom_blob_name]]
                logging.debug('{blob:s}:{scale:f}'.format(blob=bottom_blob_name, scale=base_scale))
                if is_convolution_layer(layer.type):
                    base_scale *= get_conv_weight_scale(layer.convolution_param,\
                                                        False) * \
                                    get_conv_internal_scale(layer.convolution_param, False)
                    base_offset = 0
                elif is_inner_product_layer(layer.type):
                    base_scale *= get_conv_weight_scale(layer.inner_product_param,\
                                                        True) *\
                                  get_conv_internal_scale(layer.inner_product_param,\
                                                        True) 
                    base_offset = 0
                elif is_convertor_layer(layer.type):
                    base_offset = convertor_offset( layer.convert_param, base_scale, base_offset)
                    base_scale *= convertor_scaling(layer.convert_param)
                elif is_lut_layer(layer.type):
                    lut_param = caffe_pb2.LUTParameter

                    if is_lrn_layer(layer.type):
                        lut_param = layer.nv_lrn_param.lut_param
                    elif is_sigmoid_layer(layer.type):
                        lut_param = layer.sigmoid_param.lut_param
                    else:
                        # tanh
                        lut_param = layer.tanh_param.lut_param

                    cvt_param = lut_param.lut_convert.to_coef
                    if is_lrn_layer(layer.type):
                        base_scale *= cvt_param.scale*cvt_param.post_scale/pow(2,\
                                     cvt_param.shifter)
                    else:
                        base_scale = cvt_param.scale*cvt_param.post_scale/pow(2,\
                                     cvt_param.shifter)
                    base_offset = 0
                elif is_software_layer(layer.type):
                    base_scale  = 1
                    base_offset = 0
                elif layer.type == 'SDPX':
                    if layer.sdp_x_param.HasField('alu_type'):
                        alu_scale = 1
                        if layer.sdp_x_param.HasField('alu_cvt'):
                            alu_scale = convertor_scaling(layer.sdp_x_param.alu_cvt)
                        alu_scale *= pow(2, layer.sdp_x_param.alu_shifter)
                        assert alu_scale == base_scale

                    if layer.sdp_x_param.HasField('mul_type') and layer.sdp_x_param.mul_op == caffe_pb2.SDPXParameter.MUL:
                        if layer.sdp_x_param.HasField('mul_cvt'):
                            base_scale *= convertor_scaling(layer.sdp_x_param.mul_cvt)
                        base_scale /= pow(2, layer.sdp_x_param.mul_truncate)
                    base_offset = 0
            else:
                logging.info("Error: no upstream layer for blob:{blob:s} @ {layer:s}".format(
                        blob = bottom_blob_name, layer=layer_name))
                assert False

            logging.debug("layer:{layer:s}, scale:{scale:f} offset:{offset:f}".format(
                layer=layer_name, scale=base_scale, offset=base_offset))
            model_scale[layer_name] = base_scale
            model_offset[layer_name] = base_offset
        else:
            model_scale[layer_name] = 1.0
            model_offset[layer_name]= 0

        # Build the mapping between blob name to layer
        if len(layer.bottom) > 0:
            for bottom in layer.bottom:
                bottom_blob_name = bottom.replace(r'/', '_')
                bottom_blob_name = bottom_blob_name.replace('-', '_')
                bottom2layer[bottom_blob_name] = layer_name

        if len(layer.top) > 0:
            top_blob_name = layer.top[0].replace(r'/', '_')
            top_blob_name = top_blob_name.replace('-', '_')
            top2layer[top_blob_name] = layer_name
            logging.debug('{blob:s} <-> {layer:s}'.format(blob=top_blob_name, layer=layer_name))

    return

def is_layer_contained(model, layer_name):
    is_found = False
    for layer in model.layer:
        model_layer_name = layer.name.replace(r'/', '_')
        model_layer_name = model_layer_name.replace(r'-', '_')
        if model_layer_name == layer_name:
            is_found = True
            break;

    return is_found


def read_model_params( model, model_scale, model_offset, protofile):
    # Read the existing address book.
    try:
        f = open(protofile, "rb")
        proto_text.Merge(f.read(), model)
        f.close()
    except IOError:
        logging.error(": Could not open file.  Creating a new one.")

    update_scale(model_scale, model_offset, model)

    return

def rescale_data( data_base, scale, offset, golden_base ):
    blob_type = "top"
    for layer_name in data_base.keys():
        if layer_name in scale.keys():
            for blob_name in data_base[layer_name][blob_type].keys():
                orig_value = data_base[layer_name][blob_type][blob_name]['data_average']
                logging.debug("layer:{layer:s}, blob:{blob:s}".format(layer=layer_name, blob=blob_name))

                data_base[layer_name][blob_type][blob_name]['data_average'] /= scale[layer_name]
                data_base[layer_name][blob_type][blob_name]['data_average'] += offset[layer_name]
                if layer_name in golden_base.keys():
                    logging.debug('yilinz: layer: {layer:s}, scale: {scale:f}, offset: {offset:f}, observed: {val:f}, rescaled: {rescaled:f}, golden: {golden:f}'.format(\
                            layer=layer_name, scale=scale[layer_name], offset = offset[layer_name], val=orig_value,\
                            rescaled=data_base[layer_name][blob_type][blob_name]['data_average'],\
                            golden=golden_base[layer_name][blob_type][blob_name]['data_average']))
                else:
                    logging.debug('yilinz: layer: {layer:s}, scale: {scale:f}, offset: {offset:f}, observed: {val:f}, rescaled: {rescaled:f}'.format(\
                            layer=layer_name, scale=scale[layer_name], offset = offset[layer_name], val=orig_value,\
                            rescaled=data_base[layer_name][blob_type][blob_name]['data_average']))

    return

def main(argc, argv):
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=_description)
    parser.add_argument('--version', action="version", version=__version__)

    parser.add_argument("--dir1",
            action="store",
            dest="dir1",
            required=True,
            help="Directory1 which contains the simulation log")

    parser.add_argument("--cfg1",
            action="store",
            dest="cfg1",
            required=True,
            help="The configuration file used to run simulation 1")

    parser.add_argument("--dir2",
            action="store",
            dest="dir2",
            required=True,
            help="Directory2 which contains the simulation log")

    parser.add_argument("--cfg2",
            action="store",
            dest="cfg2",
            required=True,
            help="The configuration file used to run simulation 2")

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

    model_params1           = caffe_pb2.NetParameter()
    model_scale1            = dict_of_dict()
    model_offset1           = dict_of_dict()
    dict_data1              = dict_of_dict()
    model_params2           = caffe_pb2.NetParameter()
    model_scale2            = dict_of_dict()
    model_offset2           = dict_of_dict()
    dict_data2              = dict_of_dict()

    # Get model parameters
    read_model_params( model_params1, model_scale1, model_offset1, _args.cfg1)
    read_model_params( model_params2, model_scale2, model_offset2, _args.cfg2)

    calc_data_average(dict_data1, _args.dir1)
    calc_data_average(dict_data2, _args.dir2)
  
    rescale_data(dict_data1, model_scale1, model_offset1, dict_data1 )
    rescale_data(dict_data2, model_scale2, model_offset2, dict_data2 )

    get_delta(dict_data1, dict_data2, model_scale1, model_offset1, model_params1, model_scale2, model_offset2)

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
