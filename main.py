from sklearn.model_selection._split import train_test_split

from classify_data import *
from detector import Detector
from write_solution import *


def preprocessor(image):
	levels = 3
	bins = 10

	hsv_image = rgb2hsv(image)
	hue_histogram = np.histogram(hsv_image[:, :, 0].flatten(), bins)[0]
	feature_pyramid = preprocess_image(hsv_image[:, :, 2], 0, levels)

	return np.hstack([feature_pyramid, hue_histogram])


def augment_stream(stream):
	for label, image in stream:
		yield label, image  # always yield this stuff

		if label == 1:  # if we've got a positive sample, yield the mirrored image too
			yield label, np.fliplr(image)


def main():
	data_stream = stream_load_data(1000)
	data_stream = augment_stream(data_stream)
	data_stream = preprocess_stream(data_stream, preprocessor)

	X, Y = stream_to_lists(data_stream)  # load(2000, preprocessor)

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=4)
	train(X_train, y_train)

	test_faces, names = load_images_in_path("data_sync/")
	print(classifier.predict_proba([preprocessor(f) for f in test_faces]), names)

	evaluate_performance_FPFN(X_test, y_test)

	print('Train finished')

	image = load_big_image(1)

	d = Detector(classifier, preprocessor)
	boxes = d.detect(image)


# write_output_detection(image, boxes)
#
# # evaluate_performance(X,Y)
#
# big_images = load_big_images()
# write_output_detection(big_images, boxes)


main()
