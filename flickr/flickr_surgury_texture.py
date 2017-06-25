import caffe
import numpy as np

mlp = caffe.Net('flickr_mlp_model.prototxt', 'models/flickr_mlp_iter_180000.caffemodel', caffe.TEST)

#Texture
net = caffe.Net('flickr_texture_model.prototxt', 'models/VGG_ILSVRC_19_layers.caffemodel', caffe.TEST)

net.params['scale'][0].data[...] = mlp.params['scale'][0].data[...]
net.params['scale'][1].data[...] = mlp.params['scale'][1].data[...]
net.params['gram_inner1'][0].data[...] = mlp.params['gram_inner1'][0].data[...]
net.params['gram_inner1'][1].data[...] = mlp.params['gram_inner1'][1].data[...]
net.params['gram_inner2'][0].data[...] = mlp.params['gram_inner2'][0].data[...]
net.params['gram_inner2'][1].data[...] = mlp.params['gram_inner2'][1].data[...]

net.save('models/flickr_texture_pretrain.caffemodel')
