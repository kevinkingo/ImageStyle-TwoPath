
../caffe/build/tools/caffe train -gpu 0,1 -solver flickr_mlp_solver.prototxt 2>&1 | tee ./logs/flickr_mlp_log.txt

