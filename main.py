from classify_data import *
from detector import Detector
from write_solution import *


def preprocessor(image):
	levels = 3
	hsv_image = rgb2hsv(image)

	return preprocess_image(hsv_image[:, :, 2], 0, levels)


def main():
	data_stream = stream_load_data(1000)
	data_stream = augment_stream(data_stream)
	data_stream = preprocess_stream(data_stream, preprocessor)

	X, Y = stream_to_lists(data_stream)  # load(2000, preprocessor)

	train(X, Y)
	evaluate_performance_FPFN(X, Y)

	print('Train finished')

	image = load_big_image(0)

	d = Detector(classifier, preprocessor)
	boxes, image = d.detect(image)

	write_output_detection(image, boxes)

	# evaluate_performance(X,Y)



	big_images = load_big_images()
	write_output_detection(big_images, boxes)


main()
