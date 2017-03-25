import numpy as np


def preprocess(X, num_levels=3, start_level=0):
    return [preprocess_image(i, start_level, start_level + num_levels) for i in X]


def preprocess_image(image, min_level, max_level):
    features = np.ndarray((0))

    image_size = image[0].shape[0]

    for level in range(min_level, max_level):
        num_patches = 2 ** level
        patch_size = image_size // num_patches

        for i in range(num_patches):
            for j in range(num_patches):
                x = patch_size * i
                y = patch_size * j

                filtered = extract_features(image[x:x + patch_size, y:y + patch_size])
                features = np.concatenate((features, filtered))

    return features


def extract_features(img):
    (w, h) = img.shape

    ft = [np.sum(img[:w // 2, :]) - np.sum(img[w // 2:, :])]
    ft.append(np.sum(img[:, :h // 2]) - np.sum(img[:, h // 2:]))

    sl = np.sum(img[:w // 3, :]) - np.sum(img[w // 3:w // 3 * 2, :]) + np.sum(img[w // 3 * 2:, :])
    ft.append(sl)

    sl = np.sum(img[:, :h // 3]) - np.sum(img[:, h // 3:h // 3 * 2]) + np.sum(img[:, h // 3 * 2:])
    ft.append(sl)

    sl = np.sum(img[:w // 2, :h // 2]) + np.sum(img[w // 2:, h // 2:])
    sl -= np.sum(img[w // 2:, :h // 2]) + np.sum(img[:w // 2, h // 2:])

    ft.append(sl)

    return np.array(ft)
