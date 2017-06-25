
../caffe/build/tools/caffe train -gpu 0,1 -solver ava_object_solver.prototxt -weights models/VGG_ILSVRC_19_layers.caffemodel 2>&1 | tee ./logs/ava_object_log.txt
