import numpy as np


def preprocess(X, num_levels=3, start_level=0):
	return [preprocess_image(i, start_level, start_level + num_levels) for i in X]


def preprocess_image(image, min_level, max_level):
	features = np.ndarray((0))

	image_size = image.shape[0]

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

def preprocess_image_partial(image, min_level, max_level):
	return preprocess_partial(np.cumsum(np.cumsum(image, axis=0), axis=1), min_level, max_level)

def preprocess_partial(partial, min_level, max_level):
	features = np.ndarray((0))

	image_size = partial[0].shape[0]

	num_pixels = (image_size * image_size)

	for level in range(min_level, max_level):
		num_patches = 2 ** level
		patch_size = image_size // num_patches

		for i in range(num_patches):
			for j in range(num_patches):
				x = patch_size * i
				y = patch_size * j

				filtered = extract_features_partial(partial, x, y, patch_size) / num_pixels

				features = np.concatenate((features, filtered))

	return features


def extract_features_partial(part, x0, y0, patch_size):
	def sum_part(part, x1, y1, x2, y2):
		x1 += x0
		x2 += x0
		y1 += y0
		y2 += y0
		return part[x1, y1] + part[x2, y2] - part[x1, y2] - part[x2, y1]

	(w, h) = (patch_size, patch_size)  # part.shape

	w -= 1
	h -= 1

	w2 = w // 2
	w3 = w // 3
	h2 = h // 2
	h3 = h // 3

	ft = []

	ft.append(sum_part(part, 0, 0, w2, h) - sum_part(part, w2, 0, w, h))
	ft.append(sum_part(part, 0, 0, w, h2) - sum_part(part, 0, h2, w, h))

	ft.append(sum_part(part, 0, 0, w3, h) - sum_part(part, w3, 0, w3 * 2, h) + sum_part(part, w3 * 2, 0, w, h))
	ft.append(sum_part(part, 0, 0, w, h3) - sum_part(part, 0, h3, w, h3 * 2) + sum_part(part, 0, h3 * 2, w, h))

	ft.append(sum_part(part, 0, 0, w2, h2) + sum_part(part, w2, h2, w, h) -
			  sum_part(part, w2, 0, w, h2) - sum_part(part, 0, h2, w2, h))

	return np.array(ft)


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
