qsub --projectMode direct -P mobile_t194_hw_mmplex_nvdla -q o_cpu_4G -m rel6 -Is \
    gdb ./build/tools/caffe


##    -nvconfig nv_config/caffe_int_8_hybrid.txt \
##    -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel \
##    -nvconfig nv_config/caffe_fp_16.txt \
##    -nvconfig nv_config/caffe_fp_16_bias.txt \
##    -nvconfig nv_config/caffe_int_8.txt \
##    -nvconfig nv_config/caffe_int_16.txt \
##    -nvconfig nv_config/caffe_fixed_8.txt \
##    -nvconfig nv_config/caffe_fixed_16.txt \
##    -nvconfig nv_config/caffe_fp_32.txt \
#   set args test -model models/bvlc_alexnet/yilinz_train_val.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel  -nvconfig nv_config/caffe_int_8_hybrid.txt
#   set args test -model models/bvlc_alexnet/debug_train_val.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel  -nvconfig nv_config/caffe_int_8_hybrid.txt
#   set args test -model models/bvlc_alexnet/train_val.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel  -nvconfig nv_config/caffe_int_8_hybrid.txt
#   set args test -model models/bvlc_alexnet/amod_full_test.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel  -nvconfig nv_config/caffe_int_8_hybrid.txt
#   set args test -model testcase/bvlc_alexnet/convert_int8_stress_test.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model testcase/bvlc_alexnet/convert_int32_stress_test.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model testcase/bvlc_alexnet/splitc_test.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model analysis/bvlc_alexnet/stats_val.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model analysis/bvlc_alexnet/hw_8b_hybrid_cc_sdp32_splitc32_wg10.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model analysis/bvlc_alexnet/hw_8b_hybrid_cc_sdp32_splitc32_wg_f10_w11.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model analysis/bvlc_alexnet/hw_8b_hybrid_cc_sdp32_splitc32_wg_f10_w10_lrn_seperate_rawrange.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model analysis/bvlc_alexnet/hw_8b_hybrid_cc_sdp32_splitc32_wg_f10_w10_lrn_densize16.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model models/bvlc_alexnet/train_val_autogen_wg.prototxt -iterations 1 -skip 0 -process 0 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model models/bvlc_alexnet/train_val.prototxt -iterations 1 -skip 0 -process 0 -histfiles 25 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model models/bvlc_alexnet/train_val.prototxt -iterations 1 -skip 0 -process 0 -histfiles 50 -calib 1 -pipeline 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model models/bvlc_alexnet/train_val.prototxt -iterations 1 -skip 0 -process 0 -histfiles -1 -calib 0 -pipeline 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model analysis/bvlc_alexnet/hw_8b_hybrid_cc_sdp32_splitc32_wg_f10_w8_lrn_seperate_rawrange.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model analysis/bvlc_alexnet/hw_8b_hybrid_cc_sdp32_splitc32_nowg_lrn_seperate_rawrange.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model analysis/bvlc_alexnet/hw_8b_hybrid_cc_sdp32_splitc32_wg_f10_w_double_lrn_54dbl.prototxt -iterations 1 -weights models/bvlc_alexnet/bvlc_alexnet.caffemodel
#   set args test -model models/bvlc_googlenet/train_val_test.prototxt -iterations 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -histfiles 0 -process 0 -skip 0
#   set args test -model models/bvlc_googlenet/train_val_autogen_test_32768.prototxt -iterations 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -histfiles 0 -process 0 -skip 0
#   set args test -model models/bvlc_googlenet/train_val_autogen_test.prototxt -iterations 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -histfiles 0 -process 0 -skip 0
#   set args test -model models/bvlc_googlenet/train_val_autogen.prototxt -iterations 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -histfiles 0 -process 0 -skip 0
#   set args test -model models/bvlc_googlenet/train_val.prototxt -iterations 1 -skip 0 -process 0 -histfiles 200 -calib 1 -pipeline 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel
#   set args test -model models/bvlc_googlenet/train_val_autogen.prototxt -iterations 1 -skip 0 -process 0 -pipeline 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel
#   set args test -model models/bvlc_googlenet/train_val_hasaccmu.prototxt -iterations 1 -skip 0 -process 0 -pipeline 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel
#   set args test -model ../tuning_logs/accmu/accmu_64/train_val.prototxt -iterations 1 -skip 0 -process 0 -pipeline 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel
#   set args test -model analysis/power_test/train_val_tuned.prototxt -iterations 1 -skip 0 -process 0 -pipeline 1 -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -batch 1
#   set args test -model analysis/elvis_cmod/amod_cc_sample.prototxt -iterations 1
