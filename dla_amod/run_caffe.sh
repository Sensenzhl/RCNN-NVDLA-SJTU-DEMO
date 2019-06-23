#!/bin/sh

#qsub --projectMode direct -P mobile_t186_hw_mmplex_secip -q o_cpu_16G -m rel68 -Is \
#    ./build/tools/caffe test \
#    -model models/bvlc_alexnet/train_val.prototxt \
#    -iterations 1 \
#    -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel

#qsub --projectMode direct -P mobile_t186_hw_mmplex_secip -q o_cpu_16G -m rel68 -Is \
#    ./build/tools/caffe test -model models/bvlc_googlenet/train_val_noaccmu.prototxt -iterations 1 -skip 0 -process 0 -pipeline 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel
#qsub --projectMode direct -P mobile_t186_hw_mmplex_secip -q o_cpu_16G -m rel68 -Is \
#    ./build/tools/caffe test -model analysis/offset/train_val.prototxt -iterations 1 -skip 1 -interval 2 -process 0 -pipeline 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -batch 1
#qsub --projectMode direct -P mobile_t186_hw_mmplex_secip -q o_cpu_16G -m rel68 -Is \
#    ./build/tools/caffe test -model analysis/googlenet_calibration_w_offset/train_val_autogen.prototxt -iterations 1 -skip 1 -interval 2 -process 0 -pipeline 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -batch 1 | tee analysis/offset/w_offset.txt
#qsub --projectMode direct -P mobile_t186_hw_mmplex_secip -q o_cpu_16G -m rel68 -Is \
#    ./build/tools/caffe test -model analysis/googlenet_calibration_nooffset/train_val_autogen.prototxt -iterations 1 -skip 1 -interval 2 -process 0 -pipeline 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -batch 1 | tee analysis/offset/nooffset.txt

#./build/tools/caffe test -model models/bvlc_alexnet/train_val_dump.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel | tee alexnet.log
#./build/tools/caffe test -model models/colorize/train_val_dump.prototxt -iterations 1 -weights models/colorize/colorization_release_v2.caffemodel | tee colorize.log 
#./build/tools/caffe test -model models/bvlc_googlenet/train_val.prototxt -iterations 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel | tee googlenet.log 
#./build/tools/caffe test -model models/bvlc_alexnet/train_val_autogen.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel -skip 0 -batch 1 | tee log.txt
#gdb --args ./build/tools/caffe test -model models/bvlc_alexnet/train_val_autogen.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel -skip 0 -batch 1 
#qsub --projectMode direct -P mobile_t186_hw_mmplex_secip -q o_cpu_4G -m rel68 -Is \
#    ./build/tools/caffe test -model models/bvlc_alexnet/train_val_autogen.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel

#qsub --projectMode direct -P mobile_t186_hw_mmplex_secip -q o_cpu_16G -m rel68 -Is \
#    ./build/tools/caffe test -model models/colorize/train_val.prototxt -iterations 1 -weights models/colorize/colorization_release_v2.caffemodel | tee colorize.txt
#qsub --projectMode direct -P mobile_t186_hw_mmplex_secip -q o_cpu_16G -m rel68 -Is \
#    ./build/tools/caffe test -model models/fake_model/train_val_tuned.prototxt -iterations 1 -batch 1 | tee fake_model.txt 

#qsub --projectMode direct -P mobile_t186_hw_mmplex_secip -q o_cpu_16G -m rel68 -Is \
#    ./build/tools/caffe test -model analysis/precision_summary/train_val_autogen_fpmacout.prototxt -iterations 1 -skip 237 -interval 2 -process 0 -pipeline 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -batch 1 | tee analysis/fpmacout.txt
#mv loss3_top_1_0.txt loss3_top_1_0_fpmacout.txt
#qsub --projectMode direct -P mobile_t186_hw_mmplex_secip -q o_cpu_16G -m rel68 -Is \
#    ./build/tools/caffe test -model analysis/precision_summary/train_val_autogen_wg_f10b.prototxt -iterations 1 -skip 237 -interval 2 -process 0 -pipeline 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -batch 1 | tee analysis/f10b.txt
#mv loss3_top_1_0.txt loss3_top_1_0_f10b.txt

#qsub --projectMode direct -P mobile_t186_hw_mmplex_secip -q o_cpu_16G -m rel68 -Is \
#    ./build/tools/caffe test -model analysis/train_val_dbg.prototxt -iterations 1 -skip 237 -interval 2 -process 0 -pipeline 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -batch 1 | tee analysis/golden.txt

#qsub --projectMode direct -P mobile_t186_hw_mmplex_secip -q o_cpu_16G -m rel68 -Is \
#        /home/scratch.yilinz_t19x/t19x/caffe-master-amodel/build/tools/caffe test -model /home/scratch.yilinz_t19x/t19x/caffe-master-amodel/../tuning_logs/precision_cvt_autogen_wg_f10b/train_val_tuned.prototxt -iterations 1 -weights /home/scratch.yilinz_t19x/t19x/caffe-master-amodel/models/bvlc_googlenet/bvlc_googlenet.caffemodel -histfiles 0 -batch 25 -interval 2 -skip 1 -process 0 -dbfile /home/scratch.yilinz_t19x/t19x/caffe-master-amodel/../val_db0

#qsub --projectMode direct -P mobile_t186_hw_mmplex_secip -q o_cpu_16G -m rel68 -Is \
#    ./build/tools/caffe test -model models/resnet_50/ResNet-50-deploy.prototxt -iterations 1 -skip 1 -interval 2 -process 0 -pipeline 1 -weights models/resnet_50/ResNet-50-model.caffemodel -batch 1 | tee analysis/resnet_50.txt

#qsub --projectMode direct -P mobile_t186_hw_mmplex_secip -q o_cpu_16G -m rel68 -Is \
#    ./build/tools/caffe test -model models/PilotNet/elu.prototxt -iterations 1 -skip 1 -interval 2 -process 0 -pipeline 1 -weights models/PilotNet/elu.caffemodel -batch 1 | tee analysis/elu.txt
##    -nvconfig nv_config/caffe_int_8_hybrid.txt \
##    -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel \
##    -nvconfig nv_config/caffe_fp_16.txt \
##    -nvconfig nv_config/caffe_fp_16_bias.txt \
##    -nvconfig nv_config/caffe_int_8.txt \
##    -nvconfig nv_config/caffe_int_16.txt \
##    -nvconfig nv_config/caffe_fixed_8.txt \
##    -nvconfig nv_config/caffe_fixed_16.txt \
##    -nvconfig nv_config/caffe_fp_32.txt \

workspace="/home/scratch.yilinz_t19x/git/dla_amod/"
histdir="${workspace}../amod_tests/datadistri/bvlc_alexnet/"

# run fp32 model
#./build/tools/caffe test -model models/bvlc_alexnet/train_val.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel -batch 1 >& fp32.log
#./build/tools/caffe calib -model models/bvlc_alexnet/train_val.prototxt -pipeline 1 -histfiles 500 -histdir ${histdir} -calib 3
# run calibrated model
#./build/tools/caffe test -model models/bvlc_alexnet/train_val_autogen.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel -batch 1 >& adaptive_scale.log
./build/tools/caffe test -model models/bvlc_alexnet/train_val.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel -batch 1 >& adaptive_scale.log

