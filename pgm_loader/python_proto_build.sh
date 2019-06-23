#!/bin/bash 

#QSUB_PREFIX="qsub --projectMode direct -P mobile_t194_hw_mmplex_nvdla -q o_cpu_16G -m rel6 -Is"
QSUB_PREFIX=""
$QSUB_PREFIX protoc -I=./ --python_out=./ ./caffe.proto

