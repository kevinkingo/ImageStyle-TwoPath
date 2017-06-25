
../caffe/build/tools/caffe train -gpu 0,1 -solver ava_texture_solver.prototxt -weights models/ava_texture_pretrain.caffemodel 2>&1 | tee ./logs/ava_texture_log.txt

