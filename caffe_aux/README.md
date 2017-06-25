GramLayer Installation
==================

Add GramLayer into caffe:
1. Put `gram_layer.hpp` in `caffe/include/caffe/layers/`.
1. Put `gram_layer.cpp` and `gram_layer.cu` in `caffe/src/caffe/layers/`.
1. Modify the `caffe.proto` in `caffe/src/caffe/proto/`. 
* Add the following code into file:
```
message GramParameter {
  optional uint32 down_channel = 1 [default = 32];
}
```
* Assign a number to gramlayer in `message LayerParameter`. For my version, it's:
```
optional GramParameter gram_param = 145;
```
4. Recompile caffe