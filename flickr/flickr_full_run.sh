
../caffe/build/tools/caffe train -gpu 0,1 -solver flickr_full_solver.prototxt -weights models/flickr_full_pretrain.caffemodel 2>&1 | tee ./logs/flickr_full_log.txt

