from skimage import io
from skimage.color import rgb2hsv
import os
import gc
import numpy as np

from skimage.color.colorconv import rgb2gray


def load(number = 1000):
	load_file = 'data/train_data/train_image.txt'
	label_file = 'data/train_data/train_label.txt'
	load_f = open(load_file, 'rb')
	label_f = open(label_file, 'rb')

	paths = load_f.readlines()
	labels = label_f.readlines()
	#print(paths[0])
	paths = [os.path.join(str('data/train_data/train/'),str(p.strip())[2:-1] ) for p in paths]
	labels = [int(l) for l in labels]

	#print paths


	imgs = []
	test_count = 0
	discarded = 0
	for path in paths:
		if '5203' in path or '5930' in path:
			continue
		img = io.imread(path)
		img = rgb2hsv(img)
		img = img[:,:,2]

		# img = rgb2gray(img)

		# remove images which have 45% or more of 0.51+-0.1 (green when viz hsv)
		#green = 0.51
		#mask = (img < green + 0.1) & (img > green - 0.1)
		#ratio = np.count_nonzero(mask)/(img.shape[0]*img.shape[1])
		#if ratio >= 0.35:
			# discard images that have to much hsv-green
	#		discarded += 1
	#		continue





		imgs.append(img)
		if test_count % 200 == 0:
			print('Loaded '+str(test_count)+' images')
			gc.collect()
		test_count += 1
		if test_count >= number:
			break

	print('We have discarded ', discarded, ' images')
	print('Positives: ', np.count_nonzero(labels[:test_count])/labels[:test_count])
	return imgs, labels[:test_count]
