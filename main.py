from sklearn.model_selection._split import train_test_split

from classify_data import *
from detector import Detector
from write_solution import *
import matplotlib.pyplot as plt
from load_data import stream_load_data_test

import random


def preprocessor(image):
	levels = 3
	bins = 10

	hsv_image = rgb2hsv(image)
	hue_histogram = np.histogram(hsv_image[:, :, 0].flatten(), bins, range=(0.0, 1.0), density=True)[0]
	feature_pyramid = preprocess_image(hsv_image[:, :, 2], 0, levels)

	return np.hstack([feature_pyramid, hue_histogram])


def augment_stream(stream):
	def random_shift(image):
		f = random.choice([20, 40, 60])
		h = f // 2

		offsets = [((f, 0), (h, h), (0, 0)), ((0, f), (h, h), (0, 0)),
				   ((h, h), (f, 0), (0, 0)), ((h, h), (0, f), (0, 0)),
				   ((0, f), (0, f), (0, 0)), ((0, f), (f, 0), (0, 0)),
				   ((f, 0), (f, 0), (0, 0)), ((f, 0), (0, f), (0, 0))]

		return np.pad(image, random.choice(offsets), 'constant', constant_values=(0.5,))


	for label, image in stream:
		yield label, image  # always yield this stuff

		if label == 1:
			for _ in range(4):
				yield label, random_shift(image)


def main():
	data_stream = stream_load_data(1000)
	data_stream = augment_stream(data_stream)
	data_stream = preprocess_stream(data_stream, preprocessor)

	X, Y = stream_to_lists(data_stream)  # load(2000, preprocessor)

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=4)
	print('Training...')
	train(X_train, y_train)

	# test_faces, names = load_images_in_path("data_sync/")
	# print(classifier.predict_proba([preprocessor(f) for f in test_faces]), names)

	# evaluate_performance_FPFN(X_test, y_test)

	print('Train finished')

	test_imgs_and_f_names = stream_load_data_test(1000)
	test_features = []
	test_names = []
	for t_img, f_name in test_imgs_and_f_names: 
		t = preprocessor(t_img)
		test_features.append(t)
		test_names.append(f_name)

	print('Running prediction on test data...')
	predictions = predict(test_features)

	#test_imgs_and_f_names = stream_load_data_test(50)
	#for i, (t_img, f_name) in enumerate(test_imgs_and_f_names): 
	#	plt.imshow(t_img)
	#	plt.title(predictions[i])
	#	plt.show()

	print('Saving predictions of test data...')


	write_output_classification(predictions, test_names)


	#image = load_big_image(3)
	#d = Detector(classifier, preprocessor)
	#boxes = d.detect(image)


# write_output_detection(image, boxes)
#
# # evaluate_performance(X,Y)
#
# big_images = load_big_images()
# write_output_detection(big_images, boxes)


main()
