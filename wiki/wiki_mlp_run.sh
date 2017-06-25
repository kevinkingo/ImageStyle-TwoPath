
../caffe/build/tools/caffe train -gpu 0,1 -solver wiki_mlp_solver.prototxt 2>&1 | tee ./logs/wiki_mlp_log.txt

