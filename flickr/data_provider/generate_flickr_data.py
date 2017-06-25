import caffe
import lmdb
import numpy as np
import cv2, random

savepath = '../data/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

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

style_pic = {}
train_pic = {}
validate_pic = {}
test_pic = {}
for key in range(20):
	style_pic[key] = []
	train_pic[key] = []
	validate_pic[key] = []
	test_pic[key] = []

with open('train.txt', 'r') as f:
	content = f.read()[:-1]
	for line in content.split('\n'):
		t = line.split(' ')
		style_pic[int(t[1])].append(t[0])

with open('test.txt', 'r') as f:
	content = f.read()[:-1]
	for line in content.split('\n'):
		t = line.split(' ')
		test_pic[int(t[1])].append(t[0])

for style in style_pic:
	for i in range(30):
		random.shuffle(style_pic[style])

	train_pic[style] = style_pic[style][:-800]
	validate_pic[style] = style_pic[style][-800:]


train_num = 0
validate_num = 0
test_num = 0
for style in train_pic:
	train_num += len(train_pic[style])
for style in validate_pic:
	validate_num += len(validate_pic[style])
for style in test_pic:
	test_num += len(test_pic[style])
print train_num, validate_num, test_num

crop_number = 4

# Train
count = 0
print 'Train total ' + str(train_num * crop_number)
index = sorted(range(0, train_num * crop_number), key=lambda x: random.random())
in_db = lmdb.open(savepath + 'flickr-train-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
	for style in train_pic:
		for pic in train_pic[style]:
			pic_data = scale(cv2.imread(pic))
			for _ in range(0, crop_number):
				cropped = crop_image(pic_data).transpose((2,0,1))
				im_dat = caffe.io.array_to_datum(cropped.astype('uint8'))
				im_dat.label = int(style)
				in_txn.put('{:0>10d}'.format(index.pop()), im_dat.SerializeToString())
				count += 1
				if count % 10000 == 0:
					print count
assert(count == train_num * crop_number)
in_db.close()

# Validate
count = 0
print 'Validate total ' + str(validate_num * crop_number)
index = sorted(range(0, validate_num * crop_number), key=lambda x: random.random())
in_db = lmdb.open(savepath + 'flickr-validate-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
	for style in validate_pic:
		for pic in validate_pic[style]:
			pic_data = scale(cv2.imread(pic))
			for _ in range(0, crop_number):
				cropped = crop_image(pic_data).transpose((2,0,1))
				im_dat = caffe.io.array_to_datum(cropped.astype('uint8'))
				im_dat.label = int(style)
				in_txn.put('{:0>10d}'.format(index.pop()), im_dat.SerializeToString())
				count += 1
				if count % 10000 == 0:
					print count
assert(count == validate_num * crop_number)
in_db.close()

# Test
count = 0
print 'Test total ' + str(test_num)
in_db = lmdb.open(savepath + 'flickr-test-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
	for style in test_pic:
		for pic in test_pic[style]:
			pic_data = scale(cv2.imread(pic))
			for _ in range(0, crop_number):
				cropped = crop_image(pic_data).transpose((2,0,1))
				im_dat = caffe.io.array_to_datum(cropped.astype('uint8'))
				im_dat.label = int(style)
				in_txn.put('{:0>10d}'.format(count), im_dat.SerializeToString())
				count += 1
				if count % 10000 == 0:
					print count
assert(count == test_num * crop_number)
in_db.close()
