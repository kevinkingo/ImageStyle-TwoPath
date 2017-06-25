
../caffe/build/tools/caffe train -gpu 0,1 -solver wiki_texture_solver.prototxt -weights models/wiki_texture_pretrain.caffemodel 2>&1 | tee ./logs/wiki_texture_log.txt

