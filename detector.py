import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2hsv 
from skimage.filters import gaussian
from skimage.transform.pyramids import pyramid_reduce
from sklearn.cluster import KMeans,AgglomerativeClustering
from test import main_remove_overlapping



class Detector:
	debug = False
	max_centers = 40

	def __init__(self, classifier, preprocessor):
		self.classifier = classifier
		self.preprocessor = preprocessor

	def detect(self, image):
		centers = self.estimate_face_positions(image)

		if self.debug:
			self.debug_image(image, centers)

		min_size = max(image.shape[0], image.shape[1]) // 30
		max_size = min_size * 3

		sizes = range(min_size, max_size, max(1, (max_size - min_size) // 8))

		boxes = []
		probabilities = []
		for c in centers:
			max_prob = 0
			box = None
			for size in sizes:
				size2 = size // 2

				# Find the upper left corner
				x = max(0, int(c[1]) - size2)
				y = max(0, int(c[0]) - size2)				

				# Move the box up / left s.t. it fits into the image
				if x + size >= image.shape[0]:
					x = image.shape[0] - size - 1
				if y + size >= image.shape[1]:
					y = image.shape[1] - size - 1

				patch = image[x:x + size, y:y + size, :]
				features = self.preprocessor(patch)
				prob = self.classifier.predict_proba(features.reshape(1, -1))[:, 1]
				if prob > max_prob:
					max_prob = prob
					box = [x, y, size, size]
			if  max_prob > 0.5:
				#print(max_prob)
				#print("Found face at {} {}".format(y, x))
				boxes.append(box)
				probabilities.append(max_prob)

		
		boxes = main_remove_overlapping(boxes,probabilities)
		if self.debug:
			self.debug_image(self.render_boxes(image, boxes), centers)

		return boxes

	@staticmethod
	def scale_image(img):
		while max(img.shape[0], img.shape[1]) > 1600:
			img = pyramid_reduce(img, 2)

		return img


	@staticmethod
	def map_to_bool_hsv(img):
		#fig,(h,a) = plt.subplots(2,1)
		img = rgb2hsv(img)
		#h.imshow(img[:,:,0])
		#s.imshow(img[:,:,1])
		#v.imshow(img[:,:,2])f
		skin_mask = np.bitwise_and(img[:,:,0] < 0.07,img[:,:,1] < 0.6)
		plt.imshow(skin_mask)
		plt.show()		
		return skin_mask
		

	@staticmethod
	def estimate_face_positions(img):
		skin_mask = Detector.map_to_bool_rgb(img)
		y, x = np.where(skin_mask)
		skin_points = np.vstack([x, y]).T
		##Random sample
		# idx = np.random.random(len(skin_points)) >= 0.0
		return Detector.kmeans(skin_points)

	@staticmethod
	def map_to_bool_rgb(img):
		#print(img.shape,img.max(),img.min())
		img = img/img.max()#gaussian(img, sigma=7.0, multichannel=True)
		#print(img.shape,img.max(),img.min())
		imgR = img[:, :, 0] * 256
		imgG = img[:, :, 1] * 256
		imgB = img[:, :, 2] * 256

		g = np.bitwise_and(imgG >= 85, imgG <= 219)
		b = np.bitwise_and(imgB >= 36, imgB <= 172)
		gb = np.bitwise_and(g, b)

		res = np.bitwise_and(imgR >= 141, gb)

		return res

	@staticmethod
	def kmeans(bool_image):
		km = KMeans(Detector.max_centers).fit(bool_image)
		return km.cluster_centers_

	@staticmethod
	def render_boxes(image, boxes):
		from skimage.draw import polygon_perimeter
		for min_x, min_y, w, h in boxes:
			r = [min_x, min_x + w, min_x + w, min_x]
			c = [min_y, min_y, min_y + h, min_y + h]
			rr, cc = polygon_perimeter(r, c, image.shape, clip=True)

			image[rr, cc] = [0.0, 128.0, 0.0]

		return image

	@staticmethod
	def debug_image(image, centers):
		plt.imshow(image)
		plt.scatter(centers[:, 0], centers[:, 1], color='red')
		plt.show()
