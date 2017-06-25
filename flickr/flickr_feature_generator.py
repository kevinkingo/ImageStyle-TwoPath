import caffe
import lmdb
import numpy as np

caffe.set_mode_gpu()
crop_number = 4
train_num = 47997 * crop_number
validate_num = 16000 * crop_number
test_num = 15999 * crop_number
batch_num = 16

#Train
net = caffe.Net('flickr_texture_model.prototxt', 'models/VGG_ILSVRC_19_layers.caffemodel', caffe.TRAIN)
print "Training feature total: " + str(train_num)
in_db = lmdb.open('data/flickr-train-feature-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
	count = 0
	while True:
		output = net.forward(blobs=['gram_concat', 'label'])
		features = output['gram_concat']
		features = features.reshape(features.shape[0], -1, 1, 1).astype(float)
		labels = output['label']

		num = min(batch_num, train_num - count)
		for i in range(num):
			im_dat = caffe.io.array_to_datum(features[i])
			im_dat.label = int(labels[i])
			in_txn.put('{:0>10d}'.format(count + i), im_dat.SerializeToString())
		count += num
		if count % 10000 == 0:
			print count
		if count >= train_num:
			break
assert(count == train_num)
in_db.close()

#Validate
net = caffe.Net('flickr_texture_model.prototxt', 'models/VGG_ILSVRC_19_layers.caffemodel', caffe.TEST)
print "Validation feature total: " + str(validate_num)
in_db = lmdb.open('data/flickr-validate-feature-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
	count = 0
	while True:
		output = net.forward(blobs=['gram_concat', 'label'])
		features = output['gram_concat']
		features = features.reshape(features.shape[0], -1, 1, 1).astype(float)
		labels = output['label']

		num = min(batch_num, validate_num - count)
		for i in range(num):
			im_dat = caffe.io.array_to_datum(features[i])
			im_dat.label = int(labels[i])
			in_txn.put('{:0>10d}'.format(count + i), im_dat.SerializeToString())
		count += num
		if count % 10000 == 0:
			print count
		if count >= validate_num:
			break
assert(count == validate_num)
in_db.close()