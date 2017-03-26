from classify_data import *
from detector import Detector
from write_solution import *

levels = 3


def preprocessor(image):
	return preprocess_image(rgb2hsv(image)[:, :, 2], 0, levels)


def main():
	image = load_big_image(0)

	X, Y = load(2000, lambda x: preprocess_image(x, 0, levels))

	train(X, Y)
	print('Train finished')

	d = Detector(classifier, preprocessor)
	boxes, image = d.detect(image)

	write_output_detection(image, boxes)

	# evaluate_performance(X,Y)



	big_images = load_big_images()
	boxes = extract_boxes(big_images)
	write_output_detection(big_images, boxes)


main()
