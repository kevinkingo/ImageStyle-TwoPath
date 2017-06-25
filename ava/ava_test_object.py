import caffe
import lmdb
import numpy as np
import m_ap

caffe.set_mode_gpu()

crop_number = 8
test_num = 2573
classify_num = 14
batch_num = 16

def softmax(x):
	e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
	return e_x / e_x.sum(axis=1, keepdims=True)

#Test
save = True
path = 'prob/'
if save:
	net = caffe.Net('ava_test_object_model.prototxt', 'models/ava_object_iter_20000.caffemodel', caffe.TEST)
	probs_object = np.zeros((test_num * crop_number, classify_num))
	labels = np.zeros((test_num * crop_number, classify_num), dtype=np.int)

	count = 0
	while True:
		num = min(batch_num, test_num * crop_number - count)
		output = net.forward(blobs=['newfc8'])
		probs_object[count:count + num, :] = output['newfc8'].reshape(batch_num, classify_num)[:num, :]

		count += num
		if count % 10000 == 0:
			print count
		if count >= test_num * crop_number:
			break
	assert(count == test_num * crop_number)

	probs_object = (probs_object[0::crop_number, :] + probs_object[1::crop_number, :] + probs_object[2::crop_number, :] + probs_object[3::crop_number, :] + probs_object[4::crop_number, :] + probs_object[5::crop_number, :] + probs_object[6::crop_number, :] + probs_object[7::crop_number, :]) / crop_number
	labels = labels[::crop_number, :]

	np.save(path + 'ava_test_object.npy', probs_object)
	np.save(path + 'ava_test_label.npy', labels)
else:
	probs_object = np.load(path + 'ava_test_object.npy')
	labels = np.load(path + 'ava_test_label.npy')



#Calculation
with open('data/test.multilab', 'r') as f:
	content = f.read().split('\n')
	for i in range(len(content)):
		flags = content[i].split(' ')
		for j in range(classify_num):
			labels[i, j] = int(flags[j])

print "Object Pathway Network"
probs = softmax(probs_object)
actual = []
predict = []
for i in range(classify_num):
	actual.append(list(np.nonzero(labels[:, i] == 1)[0]))
	predict.append( sorted(range(test_num), key=lambda x: -probs[x, i]) ) 

print "mAP"
print m_ap.mapk(actual, predict, k=1e9)
