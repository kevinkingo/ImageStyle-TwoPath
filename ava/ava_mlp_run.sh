
../caffe/build/tools/caffe train -gpu 0,1 -solver ava_mlp_solver.prototxt 2>&1 | tee ./logs/ava_mlp_log.txt

