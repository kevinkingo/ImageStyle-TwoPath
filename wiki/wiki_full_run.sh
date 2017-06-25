
../caffe/build/tools/caffe train -gpu 0,1 -solver wiki_full_solver.prototxt -weights models/wiki_full_pretrain.caffemodel 2>&1 | tee ./logs/wiki_full_log.txt

