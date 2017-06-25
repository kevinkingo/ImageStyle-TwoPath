import caffe
import lmdb
import numpy as np
import cv2, random

path = 'images/'
savepath = '../data/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

def read_data():
	labels = {}
	with open('valid_data', 'r') as f:
		content = f.read()
	for line in content.split('\n'):
		t = line.split('\t')
		labels[t[0]] = int(t[1])
	return labels

def get_crop_number():
	crop_number = {}
	with open('crop_number', 'r') as f:
		content = f.read()
	for line in content.split('\n'):
		t = line.split('\t')
		crop_number[int(t[0])] = int(t[1])
	return crop_number

def get_style_map():
	style_map = {}
	with open('style_map', 'r') as f:
		content = f.read()
	for line in content.split('\n'):
		t = line.split('\t')
		style_map[int(t[1])] = t[0]
	return style_map

labels = read_data()
crop_number = get_crop_number()
style_map = get_style_map()
crop_number_test = 4


style_pic = {}
train_pic = {}
validate_pic = {}
test_pic = {}
for key in range(25):
	style_pic[key] = []
	train_pic[key] = []
	validate_pic[key] = []
	test_pic[key] = []

for key in labels: 
	style_pic[labels[key]].append(key)

for style in style_pic:
	point1 = int(3 * len(style_pic[style]) / 5)
	point2 = int(4 * len(style_pic[style]) / 5)
	train_pic[style] = style_pic[style][0:point1]
	validate_pic[style] = style_pic[style][point1:point2]
	test_pic[style] = style_pic[style][point2:]

train_num = 0
validate_num = 0
test_num = 0

for style in train_pic:
	train_num += len(train_pic[style]) * crop_number[style]
	print style_map[style], len(style_pic[style]), len(train_pic[style]) * crop_number[style]
for style in validate_pic:
	validate_num += len(validate_pic[style]) 
for style in test_pic:
	test_num += len(test_pic[style]) * crop_number_test
print train_num, validate_num, test_num

def crop_image(img):
	crop_size = 224

	range_h = img.shape[0] - crop_size
	range_w = img.shape[1] - crop_size

	y = random.randrange(0, range_h)
	x = random.randrange(0, range_w)

	return img[y : y + crop_size, x : x + crop_size]

def scale(im):
	return cv2.resize(im, (256, 256))
	size = 256.0
	if im.shape[0] < im.shape[1]:
		r = size / im.shape[0]
		dim = (int(im.shape[1] * r), int(size))
	else:
		r = size / im.shape[1]
		dim = (int(size), int(im.shape[0] * r))
	return cv2.resize(im, dim)

# Train
count = 0
print 'Train total ' + str(train_num)
index = sorted(range(0, train_num), key=lambda x: random.random())
in_db = lmdb.open(savepath + 'wiki-train-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
	for style in train_pic:
		for pic in train_pic[style]:
			pic_data = scale(cv2.imread(path + pic + '.jpg'))
			for _ in range(0, crop_number[style]):
				cropped = crop_image(pic_data).transpose((2,0,1))
				im_dat = caffe.io.array_to_datum(cropped.astype('uint8'))
				im_dat.label = int(style)
				in_txn.put('{:0>10d}'.format(index.pop()), im_dat.SerializeToString())
				count += 1
				if count % 10000 == 0:
					print count
assert(count == train_num)
in_db.close()

# Validate
count = 0
print 'Validate total ' + str(validate_num)
index = sorted(range(0, validate_num), key=lambda x: random.random())
in_db = lmdb.open(savepath + 'wiki-validate-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
	for style in validate_pic:
		for pic in validate_pic[style]:
			pic_data = scale(cv2.imread(path + pic + '.jpg'))
			for _ in range(0, 1):
				cropped = crop_image(pic_data).transpose((2,0,1))
				im_dat = caffe.io.array_to_datum(cropped.astype('uint8'))
				im_dat.label = int(style)
				in_txn.put('{:0>10d}'.format(index.pop()), im_dat.SerializeToString())
				count += 1
				if count % 10000 == 0:
					print count
assert(count == validate_num)
in_db.close()

# Test
count = 0
print 'Test total ' + str(test_num)
in_db = lmdb.open(savepath + 'wiki-test-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
	for style in test_pic:
		for pic in test_pic[style]:
			pic_data = scale(cv2.imread(path + pic + '.jpg'))
			for _ in range(0, crop_number_test):
				cropped = crop_image(pic_data).transpose((2,0,1))
				im_dat = caffe.io.array_to_datum(cropped.astype('uint8'))
				im_dat.label = int(style)
				in_txn.put('{:0>10d}'.format(count), im_dat.SerializeToString())
				count += 1
				if count % 10000 == 0:
					print count
assert(count == test_num)
in_db.close()
