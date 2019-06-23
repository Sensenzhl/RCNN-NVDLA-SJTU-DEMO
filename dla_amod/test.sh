#! /usr/bin/env bash

#/home/scratch.yilinz_t19x/git/dla_amod/build/tools/caffe test -model /home/scratch.yilinz_t19x/git/dla_amod/models/bvlc_alexnet/train_val.prototxt -iterations 1 -weights /home/scratch.yilinz_t19x/git/dla_amod/models/bvlc_alexnet/bvlc_alexnet.caffemodel -histfiles 0 -batch 100 -interval 2  -skip 44001 -process 440 -dbfile /home/scratch.yilinz_t19x/git/dla_amod/../db/val_db8 
/home/scratch.yilinz_t19x/git/dla_amod/build/tools/caffe test -model /home/scratch.yilinz_t19x/git/dla_amod/models/bvlc_alexnet/train_val.prototxt -iterations 1 -weights /home/scratch.yilinz_t19x/git/dla_amod/models/bvlc_alexnet/bvlc_alexnet.caffemodel -histfiles 0 -interval 2  -skip 44001 -process 440 -dbfile /home/scratch.yilinz_t19x/git/dla_amod/../db/val_db8 >& ref.log
