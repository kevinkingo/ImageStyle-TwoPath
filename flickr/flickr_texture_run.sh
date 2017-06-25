
../caffe/build/tools/caffe train -gpu 0,1 -solver flickr_texture_solver.prototxt -weights models/flickr_texture_pretrain.caffemodel 2>&1 | tee ./logs/flickr_texture_log.txt

