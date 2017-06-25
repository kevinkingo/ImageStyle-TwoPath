import caffe
import lmdb
import numpy as np
import m_ap

caffe.set_mode_gpu()

crop_number = 4
test_num = 65972 / crop_number
classify_num = 25
batch_num = 64

def softmax(x):
	e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
	return e_x / e_x.sum(axis=1, keepdims=True)

#Test
save = True
path = 'prob/'
if save:
	net = caffe.Net('wiki_test_object_model.prototxt', 'models/wiki_object_iter_40000.caffemodel', caffe.TEST)
	probs = np.zeros((test_num * crop_number, classify_num))
	labels = np.zeros((test_num * crop_number, 1))
	count = 0
	while True:
		num = min(batch_num, test_num * crop_number - count)
		output = net.forward(blobs=['newfc8', 'label'])
		probs[count:count + num, :] = output['newfc8'].reshape(batch_num, classify_num)[:num, :]
		labels[count:count + num, :] = output['label'].reshape(batch_num, 1)[:num, :]

		count += num
		if count % 10000 == 0:
			print count
		if count >= test_num * crop_number:
			break
	assert(count == test_num * crop_number)

	probs = (probs[0::crop_number, :] + probs[1::crop_number, :] + probs[2::crop_number, :] + probs[3::crop_number, :]) / crop_number
	labels = labels[::crop_number, :]
	
	np.save(path + 'wiki_test_object.npy', probs)
	np.save(path + 'wiki_test_label.npy', labels)
else:
	probs = np.load(path + 'wiki_test_object.npy')
	labels = np.load(path + 'wiki_test_label.npy')




actual = []
for i in range(classify_num):
	actual.append(list(np.nonzero(labels == i)[0]))

print "Object Pathway Network"
probs = softmax(probs_object)

results = (labels == np.argmax(probs, axis=1).reshape(test_num, 1))
print "SA:"
print float(np.sum(results)) / test_num

category_accuracy = 0
for i in range(classify_num):
	mask = (labels == i)
	category_accuracy += float(np.sum(results * mask)) / np.sum(mask)
print "CA:"
print category_accuracy / classify_num

predict = []
for i in range(classify_num):
	predict.append( sorted(range(test_num), key=lambda x: -probs[x, i]) ) 
print "mAP:"
print m_ap.mapk(actual, predict, k=1e9)