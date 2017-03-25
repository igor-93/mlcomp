import numpy as np


def preprocess(X, num_levels=3, start_level=0):
	return [preprocess_image(i, start_level, start_level + num_levels) for i in X]


def preprocess_image(image, min_level, max_level):
	features = np.ndarray((0))

	image_size = image[0].shape[0]

	num_pixels = (image_size * image_size)

	for level in range(min_level, max_level):
		num_patches = 2 ** level
		patch_size = image_size // num_patches

		for i in range(num_patches):
			for j in range(num_patches):
				x = patch_size * i
				y = patch_size * j

				patch = image[x:x + patch_size, y:y + patch_size]

				filtered = extract_features(patch) / num_pixels

				features = np.concatenate((features, filtered))

	return features


def extract_features(img):
	(w, h) = img.shape

	w2 = w // 2
	w3 = w // 3
	h2 = h // 2
	h3 = h // 3

	ft = [np.sum(img[:w2, :]) - np.sum(img[w2:, :])]
	ft.append(np.sum(img[:, :h2]) - np.sum(img[:, h2:]))

	sl = np.sum(img[:w3, :]) - np.sum(img[w3:w3 * 2, :]) + np.sum(img[w3 * 2:, :])
	ft.append(sl)

	sl = np.sum(img[:, :h3]) - np.sum(img[:, h3:h3 * 2]) + np.sum(img[:, h3 * 2:])
	ft.append(sl)

	sl = np.sum(img[:w2, :h2]) + np.sum(img[w2:, h2:])
	sl -= np.sum(img[w2:, :h2]) + np.sum(img[:w2, h2:])

	ft.append(sl)

	return np.array(ft)
