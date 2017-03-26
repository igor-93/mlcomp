from skimage import io
from skimage.color import rgb2hsv, hsv2rgb
import os
import gc

import numpy as np
from skimage.transform import pyramid_reduce


def stream_load_data(number):
	load_file = 'data/train_data/train_image.txt'
	label_file = 'data/train_data/train_label.txt'

	with open(load_file, 'rb') as load_f:
		paths = load_f.readlines()

	with open(label_file, 'rb') as label_f:
		labels = label_f.readlines()

	paths = [os.path.join(str('data/train_data/train/'), str(p.strip())[2:-1]) for p in paths]

	if len(paths) > number:
		paths = paths[:number - 1]

	for i, (label, path) in enumerate(zip(labels, paths)):
		try:
			yield int(label), io.imread(path)  # , path, i
		except OSError as err:
			print("Couldn't load image {}\n Error: {}".format(path, err))

		if i % 200 == 0:
			print("Loaded {} images".format(i))


def stream_load_data_test(number):
	path = 'data/test_data/test/'
	paths = [ f for f in os.listdir(path) if
				 os.path.isfile(os.path.join(path, f)) and f.endswith('.jpg')]


	if len(paths) > number:
		paths = paths[:number - 1]

	for i, f in enumerate(paths):
		f = os.path.join(path, f)
		try:
			yield io.imread(f), f  # , path, i
		except OSError as err:
			print("Couldn't load image {}\n Error: {}".format(f, err))

		if i % 200 == 0:
			print("Loaded {} TEST images".format(i))			


def preprocess_stream(stream, extractor):
	for label, image in stream:
		yield label, extractor(image)


def stream_to_lists(stream):
	collected_list = list(stream)
	return [b for a, b in collected_list], [a for a, b in collected_list]


def load(number, preprocessing):
	load_file = 'data/train_data/train_image.txt'
	label_file = 'data/train_data/train_label.txt'

	with open(load_file, 'rb') as load_f:
		paths = load_f.readlines()

	with open(label_file, 'rb') as label_f:
		labels = label_f.readlines()

	paths = [os.path.join(str('data/train_data/train/'), str(p.strip())[2:-1]) for p in paths]

	lbls = []
	data = []
	test_count = 0
	discarded = 0

	for i, path in enumerate(paths):
		try:

			img = io.imread(path)
			# if test_green(img):
			# 	continue

			lbls.append(int(labels[i]))
		except OSError as err:
			continue

		# if lbls[-1] == 1:
		#	data.append(preprocessing(np.fliplr(img)))
		#	lbls.append(int(labels[i]))

		ft = preprocessing(img)
		# ft = np.hstack([ft,hue_histogramm(img[:,:,0],10)])

		data.append(ft)

		if test_count % 200 == 0:
			print('Loaded ' + str(test_count) + ' images')
			gc.collect()
		test_count += 1
		if test_count >= number:
			break

	print('We have discarded ', number - len(data), ' images')
	print('Positives: ', np.count_nonzero(lbls) / len(lbls))

	return data, lbls


def load_big_image(i, path='data/detection_example/example/'):
	onlyfiles = [os.path.join(path, f) for f in os.listdir(path) if
				 os.path.isfile(os.path.join(path, f)) and f.endswith('.jpg')]

	return io.imread(onlyfiles[i])


# img = rgb2hsv(img)

def load_images_in_path(path):
	onlyfiles = [os.path.join(path, f) for f in os.listdir(path) if
				 os.path.isfile(os.path.join(path, f)) and f.endswith('.jpg')]

	return [io.imread(file) for file in onlyfiles], onlyfiles


def hue_histogramm(img, bins):
	return np.histogram(img.flatten(), bins)[0]


def load_big_images(path='data/detection_example/example/'):
	onlyfiles = [os.path.join(path, f) for f in os.listdir(path) if
				 os.path.isfile(os.path.join(path, f)) and f.endswith('.jpg')]

	imgs = []

	for f in onlyfiles:
		try:
			img = io.imread(f)
			img = pyramid_reduce(img, 2)
			img = rgb2hsv(img)
		# img = img[:,:,2]
		# lbls.append(int(labels[i]))
		except OSError as err:
			continue

		imgs.append(img)

	return imgs


def test_green(hsv_img):
	img = hsv_img[:, :, 0]
	discard = False
	green = 0.51
	mask = (img < green + 0.1) & (img > green - 0.1)
	ratio = np.count_nonzero(mask) / (img.shape[0] * img.shape[1])
	if ratio >= 0.35:
		discard = True

	return discard
