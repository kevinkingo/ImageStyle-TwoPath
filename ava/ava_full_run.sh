
../caffe/build/tools/caffe train -gpu 0,1 -solver ava_full_solver.prototxt -weights models/ava_full_pretrain.caffemodel 2>&1 | tee ./logs/ava_full_log.txt

