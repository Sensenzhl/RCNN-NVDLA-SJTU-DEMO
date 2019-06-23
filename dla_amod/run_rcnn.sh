#! /usr/bin/env bash

workspace=/home/scratch.yilinz_t19x/git/dla_amod/
config="${workspace}models/bvlc_reference_rcnn_ilsvrc13/train_val_autogen.prototxt"
model="${workspace}models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel"

${workspace}/build/tools/caffe test -model ${config} -weights ${model} -iterations 1
