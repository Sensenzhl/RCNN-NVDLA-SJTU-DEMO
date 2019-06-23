#! /usr/bin/env bash

workdir=/home/scratch.yilinz_t19x/git/dla_amod

## Get golden reference log for each layer output
#${workdir}/build/tools/caffe test -model ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo.prototxt -iterations 1 -weights ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo.caffemodel 

## Generate fused network and corresponding model
#${workdir}/build/tools/caffe fuse -model ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo.prototxt -weights ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo.caffemodel

### Run the fused network
#${workdir}/build/tools/caffe test -model ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo_fused_update.prototxt -iterations 1 -weights ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo_fused.caffemodel 

# Run histogram collector
${workdir}/build/tools/caffe test -model ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo_fused_update.prototxt -iterations 1 -weights ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo_fused.caffemodel -histfiles -1 -histdir ${workdir}/../amod_tests/datadistri/drivenet_v7/
${workspace}amod_scripts/merge_hist.py --model ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo_fused_update.prototxt --file_num 1 --dir ${workdir}/../amod_tests/datadistri/drivenet_v7/
${workdir}/build/tools/caffe test -model ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo_fused_update.prototxt -iterations 1 -weights ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo_fused.caffemodel -histfiles 1 -histdir  ${workdir}/../amod_tests/datadistri/drivenet_v7/
${workspace}amod_scripts/merge_hist.py --model ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo_fused_update.prototxt --file_num 1 --dir ${workdir}/../amod_tests/datadistri/drivenet_v7/

## Calibrate(INT8)
${workdir}/build/tools/caffe calib -model ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo_fused.prototxt -weights ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo_fused.caffemodel -pipeline 1 -histfiles 1 -histdir ${workdir}/../amod_tests/datadistri/drivenet_v7/ -calib 2

## Run calibrated
${workdir}/build/tools/caffe test -model ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo_fused_autogen.prototxt -weights ${workdir}/models/drivenet_v7_apollo/drivenet_v7_apollo_fused_autogen.caffemodel -iterations 1
