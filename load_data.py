from skimage import io
from skimage.color import rgb2hsv
import os
import gc

def load():
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
	for path in paths:
		if '5203' in path or '5930' in path:
			continue
		img = io.imread(path)
		img = rgb2hsv(img)
		img = img[:,:,0]
		
		imgs.append(img)
		if test_count % 200 == 0:
			print('Loaded '+str(test_count)+' images')
			gc.collect()
		test_count += 1

	return imgs, labels
