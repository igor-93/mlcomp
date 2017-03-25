from skimage import io
from skimage.color import rgb2hsv
import os
import gc


import numpy as np

from skimage.color.colorconv import rgb2gray


def load(number, preprocessing):
	load_file = 'data/train_data/train_image.txt'
	label_file = 'data/train_data/train_label.txt'
	load_f = open(load_file, 'rb')
	label_f = open(label_file, 'rb')

	paths = load_f.readlines()
	labels = label_f.readlines()
	#print(paths[0])
	paths = [os.path.join(str('data/train_data/train/'),str(p.strip())[2:-1] ) for p in paths]
	#labels = [int(l) for l in labels]
	#print paths


	lbls = []
	data = []
	test_count = 0
	discarded = 0
	for i, path in enumerate(paths):
		try:

			img = io.imread(path)
			img = rgb2hsv(img)
			img = img[:,:,2]
			lbls.append(int(labels[i]))
		except OSError as err:
			continue
		# img = rgb2gray(img)

		# remove images which have 45% or more of 0.51+-0.1 (green when viz hsv)
		#green = 0.51
		#mask = (img < green + 0.1) & (img > green - 0.1)
		#ratio = np.count_nonzero(mask)/(img.shape[0]*img.shape[1])
		#if ratio >= 0.35:
			# discard images that have to much hsv-green
	#		discarded += 1
	#		continue

		data.append(preprocessing(img))

		if test_count % 200 == 0:
			print('Loaded '+str(test_count)+' images')
			gc.collect()
		test_count += 1
		if test_count >= number:
			break

	print('We have discarded ', discarded, ' images')
	print('Positives: ', np.count_nonzero(lbls)/len(lbls))

	return data, lbls


def load_big_images(path='data/detection_example/example/'):

	onlyfiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.jpg')]

	imgs = []

	for f in onlyfiles:
		try:
			img = io.imread(f)
			img = rgb2hsv(img)
			#img = img[:,:,2]
			#lbls.append(int(labels[i])) 
		except OSError as err:
			continue

		imgs.append(img)

	return imgs


def test_green(hsv_img):
	img = hsv_img[:,:,0]
	discard = False
	green = 0.51
	mask = (img < green + 0.1) & (img > green - 0.1)
	ratio = np.count_nonzero(mask)/(img.shape[0]*img.shape[1])
	if ratio >= 0.35:
		discard = True

	return discard
