#!/bin/sh

### Example to run AlexNet FP32 model ###

workspace="/home/scratch.yilinz_t19x/git/dla_amod/"
outdir="${workspace}temp/dla"
config="${workspace}train_val_autogen.prototxt"
model="${workspace}models/bvlc_alexnet/bvlc_alexnet.caffemodel"
maxtime=180

rm -rf ${outdir}
mkdir ${outdir}

cp ${config} ${outdir}
# Validation on ILSRC dataset (use even numbered images only)
${workspace}amod_scripts/submit_jobs.pl -model_def ${config} -pre_trained ${model} -maxtime ${maxtime} -out_dir ${outdir} -skip 0 -interval 2
# get the final top1/5
${workspace}amod_scripts/amod_result_parsing.py --log_dir ${outdir} --line 3
