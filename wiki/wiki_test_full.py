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
	net = caffe.Net('wiki_test_full_model.prototxt', 'models/wiki_full_iter_30000.caffemodel', caffe.TEST)
	probs_object = np.zeros((test_num * crop_number, classify_num))
	probs_texture = np.zeros((test_num * crop_number, classify_num))
	labels = np.zeros((test_num * crop_number, 1))
	count = 0
	while True:
		num = min(batch_num, test_num * crop_number - count)
		output = net.forward(blobs=['newfc8', 'gram_fc3', 'label'])
		probs_object[count:count + num, :] = output['newfc8'].reshape(batch_num, classify_num)[:num, :]
		probs_texture[count:count + num, :] = output['gram_fc3'].reshape(batch_num, classify_num)[:num, :]
		labels[count:count + num, :] = output['label'].reshape(batch_num, 1)[:num, :]

		count += num
		if count % 10000 == 0:
			print count
		if count >= test_num * crop_number:
			break
	assert(count == test_num * crop_number)

	probs_object = (probs_object[0::crop_number, :] + probs_object[1::crop_number, :] + probs_object[2::crop_number, :] + probs_object[3::crop_number, :]) / crop_number
	probs_texture = (probs_texture[0::crop_number, :] + probs_texture[1::crop_number, :] + probs_texture[2::crop_number, :] + probs_texture[3::crop_number, :]) / crop_number
	labels = labels[::crop_number, :]
	
	np.save(path + 'wiki_test_full_object.npy', probs_object)
	np.save(path + 'wiki_test_full_texture.npy', probs_texture)
	np.save(path + 'wiki_test_label.npy', labels)
else:
	probs_object = np.load(path + 'wiki_test_full_object.npy')
	probs_texture = np.load(path + 'wiki_test_full_texture.npy')
	labels = np.load(path + 'wiki_test_label.npy')






actual = []
for i in range(classify_num):
	actual.append(list(np.nonzero(labels == i)[0]))

print "Merged Network"
probs = ((softmax(probs_object) + softmax(probs_texture)) * 0.5)

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


