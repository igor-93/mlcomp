from skimage import io
from skimage.color import rgb2hsv, hsv2rgb
import os
import gc
import matplotlib.pyplot as plt


import numpy as np
from skimage.filters import gaussian
from skimage.transform import pyramid_reduce
from skimage.color.colorconv import rgb2gray
from sklearn.cluster import KMeans

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
			if test_green(img):
				continue
			lbls.append(int(labels[i]))
		except OSError as err:
			continue

		#if lbls[-1] == 1:
		#	data.append(preprocessing(np.fliplr(img)))
		#	lbls.append(int(labels[i]))
		
		ft = preprocessing(img[:,:,2])
		ft = np.hstack([ft,hue_histogramm(img[:,:,0],10)])

		data.append(ft)

		if test_count % 200 == 0:
			print('Loaded '+str(test_count)+' images')
			gc.collect()
		test_count += 1
		if test_count >= number:
			break

	print('We have discarded ', number - len(data), ' images')
	print('Positives: ', np.count_nonzero(lbls)/len(lbls))

	return data, lbls

def hue_histogramm(img,bins):
	return np.histogram(img.flatten(),bins)[0]

def load_big_images(path='data/detection_example/example/'):

	onlyfiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.jpg')]

	imgs = []

	for f in onlyfiles:
		try:
			img = io.imread(f)
			img =  pyramid_reduce(img,2)
			img = rgb2hsv(img)
			#img = img[:,:,2]
			#lbls.append(int(labels[i])) 
		except OSError as err:
			continue

		imgs.append(img)

	return imgs



def map_to_bool(img):
	img = hsv2rgb(img)
	img = gaussian(img, sigma=7.0)
	imgR = img[:,:,0] * 256
	imgG = img[:,:,1] *256
	imgB = img[:,:,2] *256
	g = np.bitwise_and( imgG >= 85, imgG <= 219)
	b =  np.bitwise_and( imgB >=36,  imgB <= 172)
	gb = np.bitwise_and( g,b)
	res = np.bitwise_and(imgR >=141, gb)

	#print(np.max(imgR))
	#plt.imshow(res )
	#plt.show()

	y,x = np.where(res)
	coords = np.vstack ([x,y]).T
	print(coords.shape)
	return coords


def kmeans_img(coords):
	km = KMeans(60).fit(coords)
	centers = km.cluster_centers_
	return centers


def test_green(hsv_img):
	img = hsv_img[:,:,0]
	discard = False
	green = 0.51
	mask = (img < green + 0.1) & (img > green - 0.1)
	ratio = np.count_nonzero(mask)/(img.shape[0]*img.shape[1])
	if ratio >= 0.35:
		discard = True

	return discard
