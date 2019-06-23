#!/bin/sh

### Example to run AlexNet INT8 calibration and validation on ILSVRC dataset ###

workspace="/home/scratch.yilinz_t19x/git/dla_amod/"
# uses the histogram from HK xavier
#histdir="${workspace}../amod_tests/datadistri/hk_xavier_hist/"
#outdir="${workspace}../temp/hk_xavier_hist"
# uses new collected histogram
histdir="${workspace}../amod_tests/datadistri/alexnet_shrinked_129_new_even_hist/"
outdir="${workspace}../temp/alexnet_per_atom_shifter"
config="${workspace}models/bvlc_alexnet/train_val.prototxt"
model="${workspace}models/bvlc_alexnet/bvlc_alexnet.caffemodel"
maxtime=160
hist_file_num=500

rm -rf ${outdir}
mkdir -p ${outdir}

cp ${config} ${outdir}
# Calibration
${workspace}build/tools/caffe calib -model ${config} -pipeline 1 -histfiles ${hist_file_num} -histdir ${histdir}  -calib 2 -weights ${model} >& ${outdir}/calib.log
## Validation on ILSRC dataset (use even numbered images only)
#${workspace}amod_scripts/submit_jobs.pl -model_def ${config//train_val/train_val_autogen} -pre_trained ${model} -maxtime ${maxtime} -out_dir ${outdir} -skip 1 -interval 2 -batch 1
## submit the failed tasks again
#${workspace}amod_scripts/pixiu.py --dir ${outdir} --process $hist_file_num --amod_dir ${workspace}
## get the final top1/5
#${workspace}amod_scripts/amod_result_parsing.py --log_dir ${outdir} --line 3
