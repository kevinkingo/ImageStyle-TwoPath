
../caffe/build/tools/caffe train -gpu 0,1 -solver flickr_object_solver.prototxt -weights models/VGG_ILSVRC_19_layers.caffemodel 2>&1 | tee ./logs/flickr_object_log.txt
