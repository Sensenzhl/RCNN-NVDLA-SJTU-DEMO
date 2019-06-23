#!/bin/sh

### Example to run AlexNet INT8 calibration and validation on ILSVRC dataset ###

workspace="/home/scratch.yilinz_t19x/git/dla_amod/"
histdir="${workspace}../amod_tests/datadistri/bvlc_rcnn/"
outdir="${workspace}../temp/bvlc_rcnn"
config="${workspace}models/bvlc_reference_rcnn_ilsvrc13/train_val.prototxt"
model="${workspace}models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel"
maxtime=60
hist_file_num=500

rm -rf ${outdir}
mkdir -p ${outdir}

#cp ${config} ${outdir}
## Calibration
#${workspace}build/tools/caffe calib -model ${config} -pipeline 1 -histfiles ${hist_file_num} -histdir ${histdir}  -calib 0  -weights ${model} >& ${outdir}/calib.log
## Validation on ILSRC dataset (use even numbered images only)
#${workspace}amod_scripts/submit_jobs.pl -model_def ${config//train_val/train_val_autogen} -pre_trained ${model} -maxtime ${maxtime} -out_dir ${outdir} -skip 1 -interval 2 -batch 1 -db DLA_FAST_SIMU
## submit the failed tasks again
#${workspace}amod_scripts/pixiu.py --dir ${outdir} --process $hist_file_num --amod_dir ${workspace}
## get the final top1/5
#${workspace}amod_scripts/amod_result_parsing.py --log_dir ${outdir} --line 3 --log calib_results.log
#
#outdir_fp32="${workspace}../temp/bvlc_rcnn_fp32"
#rm -rf  $outdir_fp32
#mkdir -p $outdir_fp32
#cp ${config} ${outdir_fp32}
## Calibration
## Validation on ILSRC dataset (use even numbered images only)
#${workspace}amod_scripts/submit_jobs.pl -model_def ${config} -pre_trained ${model} -maxtime ${maxtime} -out_dir ${outdir_fp32} -skip 1 -interval 2 -batch 1 -db DLA_FAST_SIMU
## submit the failed tasks again
#${workspace}amod_scripts/pixiu.py --dir ${outdir} --process $hist_file_num --amod_dir ${workspace}
## get the final top1/5
#${workspace}amod_scripts/amod_result_parsing.py --log_dir ${outdir_fp32} --line 3 --log fp32_results.log

cp ${config} ${outdir}
# Calibration for fp16
${workspace}build/tools/caffe calib -model ${config} -pipeline 0 -histfiles ${hist_file_num} -histdir ${histdir}  -calib 0  -weights ${model} >& ${outdir}/calib.log
## Validation on ILSRC dataset (use even numbered images only)
#${workspace}amod_scripts/submit_jobs.pl -model_def ${config//train_val/train_val_autogen} -pre_trained ${model} -maxtime ${maxtime} -out_dir ${outdir} -skip 1 -interval 2 -batch 1 -db DLA_FAST_SIMU
## submit the failed tasks again
#${workspace}amod_scripts/pixiu.py --dir ${outdir} --process $hist_file_num --amod_dir ${workspace}
## get the final top1/5
#${workspace}amod_scripts/amod_result_parsing.py --log_dir ${outdir} --line 3 --log calib_results.log
