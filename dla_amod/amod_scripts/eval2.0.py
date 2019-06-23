#!/home/efan/scratch_sc/anaconda2/bin/python2.7

_description='''
This tool is used to evalute the perf of nvdla2.0
'''

__version__ = '0.5'

#TODO
#1. calculate the total kernel size of DriveNet_v7 to see whether they can be filled in UBUF.

#Modes
#1. DC
#2. Winograd
#3. Winograd+Dilation
#4. Image-in

#list of result 
RESULT_ATOMIC_C_IDX = 0 
RESULT_ATOMIC_K_IDX = 1
RESULT_CONV_N_IDX   = 2 
RESULT_CONV_K_IDX   = 3 
RESULT_FEATURE_SIZE_IDX = 4
RESULT_KERNEL_SIZE_IDX = 5 
RESULT_CBUF_FEATURE_SIZE_IDX = 6 
RESULT_CBUF_WEIGHT_SIZE_IDX = 7 
RESULT_REQUIRED_CBUF_SIZE_IDX = 8 
RESULT_CPIPE_OPERATION_CYCLES_IDX = 9 
RESULT_CONV_C_IDX = 10
RESULT_CONV_C_SIZE_IDX = 11

import os
import inspect
import sys
import argparse
import xlwt 

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
caffe_pb_dir = cmd_folder + "/python/caffe/proto/"
if caffe_pb_dir not in sys.path:
    sys.path.insert(0, caffe_pb_dir)
caffe_dir = cmd_folder + "/python"
if caffe_dir not in sys.path:
    sys.path.insert(0, caffe_dir)

proto_dir = r"/home/utils/python-protobuf-2.6.1/lib/python2.7/site-packages/"
if proto_dir not in sys.path:
    sys.path.insert(0, proto_dir)

from google.protobuf import text_format as proto_text
import caffe_pb2
import caffe

#import plotly
#from plotly.graph_objs import Scatter, Layout

#Design parameters
cpipes_num = 8
cpipe_mac_cell_num = 1024
atom_operation_cycles = 1
feature_atom_read_cycles = 16
kernel_atom_read_cycles = 16
ubuf_width = 32    #Bytes
#cbuf_width = atomic_c    #Bytes
cbuf_size = 128*1024 #64KB
#ubuf_size = 4*1024*1024
#stripe_length = stripe_ratio * atomic_k #stripe_ratio is 1 or 2

#range 1
range_atomic_c = [16, 32, 64, 128]
range_conv_n = [1, 2, 4, 8]
range_conv_c_size = [16, 32, 64, 128]
range_conv_k = [1, 2, 4, 8]

##range 2
#range_atomic_c = [32, 64]
#range_conv_n = [1, 2, 4, 8]
#range_conv_c_size = [32, 64, 128]
#range_conv_k = [1, 2, 4, 8]

#Outputs
larer_cycles = 0
network_cycles = 0
required_cbuf_size = 0
assembly_buffer_size = 0
delivery_buffer_size = 0
one_cbuf_feature_read_size = 0
one_cbuf_kernel_read_size = 0
one_cbuf_write_size = 0
ubuf_read_bandwidth = 0
ubuf_write_bandwidth = 0
#cbuf_feature_size = 0
#cbuf_weight_size = 0 #double buffer

#sheet = 0

def print_layer_result(result):
    atomic_c = result[RESULT_ATOMIC_C_IDX]
    atomic_k = result[RESULT_ATOMIC_K_IDX]
    conv_n   = result[RESULT_CONV_N_IDX]
    conv_k   = result[RESULT_CONV_K_IDX]
    cbuf_feature_size  = result[RESULT_CBUF_FEATURE_SIZE_IDX]
    cbuf_weight_size   = result[RESULT_CBUF_WEIGHT_SIZE_IDX]
    required_cbuf_size = result[RESULT_REQUIRED_CBUF_SIZE_IDX]
    cpipe_operation_cycles = result[RESULT_CPIPE_OPERATION_CYCLES_IDX]
    conv_c = result[RESULT_CONV_C_IDX]
    conv_c_size = result[RESULT_CONV_C_SIZE_IDX]
    #if (required_cbuf_size <= cbuf_size):
    print('%s: atomic_c=%d atomic_k=%d conv_n=%d conv_k=%d conv_c=%d conv_c_size=%d cbuf_feature_size=%d cbuf_weight_size=%d required_cbuf_size=%d cpipe_operation_cycles=%d' % ("valid" if (required_cbuf_size <= cbuf_size) else "invalid", atomic_c, atomic_k, conv_n, conv_k, conv_c, conv_c_size, cbuf_feature_size, cbuf_weight_size, required_cbuf_size, cpipe_operation_cycles))

def read_model_params(model, protofile):
    try:
        f = open(protofile, "rb")
        proto_text.Merge(f.read(), model)
        f.close()
    except IOError:
        logging.error(": Could not open file.  Creating a new one.")
    return

def is_data_layer(layer_type):
    return layer_type == "Data"

def is_convolution_layer(layer_type):
    assert layer_type != "Deconvolution"
    return layer_type == "Convolution"

def is_pooling_layer(layer_type):
    return layer_type == "Pooling"

def calc_layers(net, model):
    global sheet
    conv_layer_idx = 0
    layer_input = {}
    for layer in model.layer:
        if is_convolution_layer(layer.type):
            conv_layer_idx = conv_layer_idx + 1
            layer_input['name'] = layer.name
            for blob in layer.bottom:
                layer_input['feature_w'] = net.blobs[blob].width
                layer_input['feature_h'] = net.blobs[blob].height
                layer_input['feature_c'] = net.blobs[blob].channels
            for blob in layer.top:
                layer_input['out_w'] = net.blobs[blob].width
                layer_input['out_h'] = net.blobs[blob].height
                layer_input['out_c'] = net.blobs[blob].channels
            conv_param = layer.convolution_param
            if len(conv_param.pad) > 0:
                pad_w = conv_param.pad[0]
                pad_h = conv_param.pad[0]
            else:
                if conv_param.pad_w > 0:
                    pad_w = conv_param.pad_w
                else:
                    pad_w = 0
                if conv_param.pad_h > 0:
                    pad_h = conv_param.pad_h
                else:
                    pad_h = 0
            layer_input['kernel_r']  = conv_param.kernel_size[0]
            layer_input['kernel_s']  = conv_param.kernel_size[0]
            layer_input['pad_w']     = pad_w
            layer_input['pad_h']     = pad_h
            if len(conv_param.stride) > 0:
                layer_input['stride_x'] = conv_param.stride[0]
                layer_input['stride_y'] = conv_param.stride[0]
            else:
                layer_input['stride_x'] = 1
                layer_input['stride_y'] = 1
            layer_input['feature_size'] = layer_input['feature_w'] * layer_input['feature_h'] * layer_input['feature_c']
            layer_input['kernel_size']  = layer_input['kernel_r'] * layer_input['kernel_s'] * layer_input['feature_c'] * layer_input['out_c']

            sheet.write(2, conv_layer_idx, layer_input['feature_w'])
            sheet.write(3, conv_layer_idx, layer_input['feature_h'])
            sheet.write(4, conv_layer_idx, layer_input['feature_c'])
            sheet.write(5, conv_layer_idx, layer_input['out_c'])
            sheet.write(6, conv_layer_idx, layer_input['out_w'])
            sheet.write(7, conv_layer_idx, layer_input['out_h'])
            sheet.write(8, conv_layer_idx, layer_input['kernel_r'])
            sheet.write(9, conv_layer_idx, layer_input['kernel_s'])
            sheet.write(10, conv_layer_idx, layer_input['stride_x'])
            sheet.write(11, conv_layer_idx, layer_input['stride_y'])
            sheet.write(12, conv_layer_idx, layer_input['pad_w'])
            sheet.write(13, conv_layer_idx, layer_input['pad_h'])
            sheet.write(14, conv_layer_idx, 1) #layer_input['dilation_x'])
            sheet.write(15, conv_layer_idx, 1) #layer_input['dilation_y'])

            calc_one_layer(conv_layer_idx, layer_input)

def calc_one_layer(conv_layer_idx, layer_input):
    global sheet
    layer_results = []
    layer_results_no_split_w = []
    layer_results_all = []

    print('layer %s, feature_size=%d kernel_size=%d' % (layer_input['name'], layer_input['feature_size'], layer_input['kernel_size']))
    print('          feature_w = %d feature_h=%d feature_c=%d' % (layer_input['feature_w'], layer_input['feature_h'], layer_input['feature_c']))
    print('          out_w= %d out_h=%d out_c=%d' % (layer_input['out_w'], layer_input['out_h'], layer_input['out_c']))
    for atomic_c in range_atomic_c:
        atomic_k = cpipe_mac_cell_num / atomic_c
        stripe_length = 2*atomic_k
        kernel_group_size = layer_input['kernel_r'] * layer_input['kernel_s'] * layer_input['feature_c'] * atomic_k
        print('   atomic_c=%d atomic_k=%d kernel_group_size=%d' % (atomic_c, atomic_k, kernel_group_size))
        for conv_n in range_conv_n:
            cpipe_out_h = (layer_input['out_h'] + conv_n - 1)/conv_n
    
            for conv_c_size in range_conv_c_size:
                #Innermost layer
                if (conv_c_size < atomic_c):
                    continue
                result = []
                for idx in range(0,100):
                    result.insert(idx, 0)
                conv_c = (layer_input['feature_c'] + conv_c_size - 1) / conv_c_size
                cpipe_out_w = layer_input['out_w']
                conv_k = cpipes_num / conv_n / conv_c 
                surf_num = (conv_c_size + atomic_c - 1)/atomic_c
                if (conv_k not in range_conv_k):
                    continue
                #cpipe_in_w = (cpipe_out_w-1) * layer_input['stride_x'] + layer_input['kernel_s']
                cpipe_in_w = layer_input['feature_w']
                cpipe_out_c = (layer_input['out_c'] + conv_k - 1)/conv_k

                group_operation_num   = (cpipe_out_c + atomic_k -1)/atomic_k
                channel_operation_num = (cpipe_out_w*cpipe_out_h + stripe_length - 1)/stripe_length
                block_operation_num = surf_num
                stripe_operation_num = layer_input['kernel_r'] * layer_input['kernel_s']
                atom_operation_num = stripe_length #in each atom operation, the number of mac is atomic_c*atomic_k
    
                result.insert(RESULT_ATOMIC_C_IDX, atomic_c)
                result.insert(RESULT_ATOMIC_K_IDX, atomic_k)
                result.insert(RESULT_CONV_N_IDX, conv_n)
                result.insert(RESULT_CONV_K_IDX, conv_k)
                result.insert(RESULT_FEATURE_SIZE_IDX, layer_input['feature_size'])
                result.insert(RESULT_KERNEL_SIZE_IDX, layer_input['kernel_size'])
                #cpipe_in_height = (cpipe_out_h-1) * layer_input['stride_y'] + layer_input['kernel_r'] - layer_input['pad_h']*2
                cpipe_in_height = (cpipe_out_h-1) * layer_input['stride_y'] + layer_input['kernel_r']
                cbuf_feature_size = cpipe_in_height * cpipe_in_w * conv_c_size
                result.insert(RESULT_CBUF_FEATURE_SIZE_IDX, cbuf_feature_size) #cbuf_feature_size
                result.insert(RESULT_CBUF_WEIGHT_SIZE_IDX, kernel_group_size*2) #cbuf_weight_size

                required_cbuf_size = cbuf_feature_size + kernel_group_size*2
                result.insert(RESULT_REQUIRED_CBUF_SIZE_IDX, required_cbuf_size)
                cpipe_operation_cycles = group_operation_num * channel_operation_num * block_operation_num * stripe_operation_num * atom_operation_num * atom_operation_cycles
                result.insert(RESULT_CPIPE_OPERATION_CYCLES_IDX, cpipe_operation_cycles)

                result.insert(RESULT_CONV_C_IDX, conv_c)
                result.insert(RESULT_CONV_C_SIZE_IDX, conv_c_size)

                #power

                print('conv_c=%d conv_c_size=%d cpipe_in_w=%d cpipe_out_h=%d cpipe_in_height=%d surf_num=%d atomic_c=%d cbuf_feature_size=%d' % (conv_c, conv_c_size, cpipe_in_w, cpipe_out_h, cpipe_in_height, surf_num, atomic_c, cbuf_feature_size))
                #print("group_operation_num=%d channel_operation_num=%d block_operation_num=%d stripe_operation_num=%d atom_operation_num=%d" %(group_operation_num, channel_operation_num, block_operation_num, stripe_operation_num, atom_operation_num))
                print_layer_result(result)

                layer_results_all.append(result)
                if (required_cbuf_size <= cbuf_size):
                    layer_results.append(result)

    #    plotly.offline.plot({
    #        "data": [Scatter(x=[1, 2, 4, 8, 16], y=[result[1]['cpipe_operation_cycles'], result[2]['cpipe_operation_cycles'], 
    #                result[4]['cpipe_operation_cycles'], result[8]['cpipe_operation_cycles'], result[16]['cpipe_operation_cycles'], ])],
    #        "layout": Layout(title='%s %s' % (layer_input['name'], 'cycles'))
    #    })
    layer_results.sort(key=lambda x:x[RESULT_CPIPE_OPERATION_CYCLES_IDX])
    layer_results_no_split_w.sort(key=lambda x:x[RESULT_CPIPE_OPERATION_CYCLES_IDX])
    print('%s sorted:' % (layer_input['name']))
    if (len(layer_results) > 0):
        for top_idx in range(0, min(len(layer_results), 5)):
            print_layer_result(layer_results[top_idx])
#        sheet.write(16, conv_layer_idx, layer_results[top_idx][RESULT_ATOMIC_C_IDX])
#        sheet.write(17, conv_layer_idx, layer_results[top_idx][RESULT_CONV_N_IDX])
#        sheet.write(18, conv_layer_idx, layer_results[top_idx][RESULT_CPIPE_OUT_W_IDX])
    else:
        print('Empty')
    print('%s sorted, no_split_w: len=%d' % (layer_input['name'], len(layer_results_no_split_w)))
    if (len(layer_results_no_split_w) > 0):
        for top_idx in range(0, min([len(layer_results_no_split_w), 5])):
            print_layer_result(layer_results_no_split_w[top_idx])
#        sheet.write(16, conv_layer_idx, layer_results_no_split_w[top_idx][RESULT_ATOMIC_C_IDX])
#        sheet.write(17, conv_layer_idx, layer_results_no_split_w[top_idx][RESULT_CONV_N_IDX])
#        sheet.write(18, conv_layer_idx, layer_results_no_split_w[top_idx][RESULT_CPIPE_OUT_W_IDX])
    else:
        print('Empty')
    print('')

#DC pipeline: (in each stripe operation)
#  CBUF+CMAC+CACC
#    read atoms from one kernel group in each position of R*S (*conv_k)
#    read feature of one stripe (*conv_n)  (in parallel: read next position of R*S, compute in mac)
#    write to CBUF
#  UBUF->CBUF
#    read feature data of one partition of feature cube (*conv_n) (it would be best if it can fill in cbuf)
#    read two kernel groups (in parallel with group operation) (*conv_k)

def main(argc, argv):
    global network_wbk
    global sheet
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=_description)
    parser.add_argument('--version', action="version", version=__version__)

    parser.add_argument("--model",
            action="store",
            dest="model",
            required=True,
            help="The network definition in *.prototxt format")

    global _args

    _args = parser.parse_args()

    net   = caffe.Net(_args.model, caffe.TEST)
    model = caffe_pb2.NetParameter()

    # Get model parameters
    read_model_params(model, _args.model)
    
    for layer in model.layer:
        print('#####' + layer.name + '####')
        for blob in layer.bottom:
            print(blob, net.blobs[blob].data.shape)
        for blob in layer.top:
            print(blob, net.blobs[blob].data.shape)

    network_wbk = xlwt.Workbook()
    sheet = network_wbk.add_sheet('tmp', cell_overwrite_ok=True)
    calc_layers(net, model)
    network_wbk.save('demo.xls')
  

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
