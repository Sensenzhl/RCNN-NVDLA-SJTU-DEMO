#!/home/yilinz/anaconda2/bin/python2.7

_description='''
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

## CONFIG START ##
AMOD_DIR    = r"/home/scratch.yilinz_t19x/git/dla_amod"
DB_DIR      = r"/home/scratch.yilinz_t19x/git/db/val_db"
MEAN_DIR    = r"/home/scratch.yilinz_t19x/git/dla_amod/data/ilsvrc12/imagenet_mean.binaryproto"
REF_DIR     = r"/home/scratch.yilinz_t19x/git/amod_tests"
## CONFIG END ##


###
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder += "/build/python/caffe/"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

class dict_of_dict(OrderedDict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return OrderedDict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


BatchNum    = 1
def main(argc, argv):
    caffe_exe = os.path.join(AMOD_DIR, r'build/tools/caffe')
    proto_dir = os.path.join(AMOD_DIR, r'testcase')
    model_dir = os.path.join(AMOD_DIR, r'models/')


    module_tests = [
        ## Convolution tests
        {'proto':r'bvlc_alexnet/splitc_test_accmu_16.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'splitc convolution test'},
        {'proto':r'bvlc_alexnet/splitc_test_accmu_32.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'splitc convolution test'},
        {'proto':r'bvlc_alexnet/splitc_test_accmu_64.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'splitc convolution test'},
        {'proto':r'bvlc_alexnet/splitc_test_accmu_inf.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'splitc convolution test'},
        {'proto':r'bvlc_alexnet/splitc_test_optimize_16.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'splitc convolution test'},
        {'proto':r'bvlc_alexnet/splitc_test_optimize_32.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'splitc convolution test'},
        {'proto':r'bvlc_alexnet/splitc_test_optimize_64.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'splitc convolution test'},
        {'proto':r'bvlc_alexnet/splitc_test_optimize_inf.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'splitc convolution test'},
        {'proto':r'bvlc_alexnet/winograd_test_accmu_16.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'winograd convolution test'},
        {'proto':r'bvlc_alexnet/winograd_test_accmu_32.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'winograd convolution test'},
        {'proto':r'bvlc_alexnet/winograd_test_accmu_64.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'winograd convolution test'},
        {'proto':r'bvlc_alexnet/winograd_test_accmu_inf.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'winograd convolution test'},
        {'proto':r'bvlc_alexnet/winograd_test_optimize_16.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'winograd convolution test'},
        {'proto':r'bvlc_alexnet/winograd_test_optimize_32.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'winograd convolution test'},
        {'proto':r'bvlc_alexnet/winograd_test_optimize_64.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'winograd convolution test'},
        {'proto':r'bvlc_alexnet/winograd_test_optimize_inf.prototxt', 'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel', 'desc':'winograd convolution test'},
        {'proto':r'conv/train_val.prototxt',                'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel',    'desc':'winograd & splitc test, failed'},
        ## Dilation & Deconvolution tests
        {'proto':r'colorize/train_val.prototxt', 'model':r'colorize/colorization_release_v2.caffemodel', 'desc':'dilation and deconvolution test'},
        ## SDP functional tests
        {'proto':r'sdp/train_val_bias.prototxt',            'model':r'bvlc_alexnet/bvlc_alexnet.caffemodel',    'desc':'SDP bias addition test'},
        {'proto':r'sdp/train_val_batchnorm.prototxt',       'model':r'resnet_152/ResNet-152-model.caffemodel',  'desc':'SDP Batch Norm test'},
        {'proto':r'sdp/train_val_prelu.prototxt',           'model':r'',  'desc':'SDP PRELU test'},
        {'proto':r'sdp/train_val_eltwise_max.prototxt',     'model':r'resnet_152/ResNet-152-model.caffemodel',  'desc':'SDP Eltwise Max test'},
        {'proto':r'sdp/train_val_eltwise_sum.prototxt',     'model':r'resnet_152/ResNet-152-model.caffemodel',  'desc':'SDP Eltwise SUM test'},
        {'proto':r'sdp/train_val_eltwise_prod.prototxt',    'model':r'resnet_152/ResNet-152-model.caffemodel',  'desc':'SDP Eltwise SUM test'},
    ]

    cmd = 'rm -rf testcase/log/'
    os.system(cmd)
    cmd = 'mkdir -p testcase/log'
    os.system(cmd)

    for test in module_tests:
        log = test['proto'].replace(r'/', '_').replace('prototxt', 'log')
        if test['model'] == '':
            cmd = caffe_exe + ' test -model ' + os.path.join(proto_dir, test['proto']) + \
            ' -iterations 1 -batch ' + str(BatchNum) + \
            ' -ref_dir ' + REF_DIR + \
            ' -dbfile ' + DB_DIR + ' -meanfile ' + MEAN_DIR + ' >& ' + os.path.join('testcase/log', log)
        else:
            cmd = caffe_exe + ' test -model ' + os.path.join(proto_dir, test['proto']) + \
            ' -iterations 1 -batch ' + str(BatchNum) +  \
            ' -ref_dir ' + REF_DIR + \
            ' -dbfile ' + DB_DIR + ' -meanfile ' + MEAN_DIR + '  -weights ' + \
            os.path.join(model_dir, test['model']) + ' >& ' + os.path.join('testcase/log', log)
        print(cmd)
        os.system(cmd);


    print("#####################################")
    print("##            Test Result          ##")
    print("#####################################")
    cmd = r'grep -E -i "fail|error" testcase/log/*.log'
    os.system(cmd)


    return


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
