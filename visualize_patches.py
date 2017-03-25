import numpy as np


def features2image2(features, level, dimensions=200):
	image = features2image(features, 1, dimensions)

	for l in range(1, level):
		image += features2image(features, l, dimensions)

	return image


def features2image(features, level, dimensions=200):
	level -= 1

	image = np.zeros((dimensions, dimensions))

	num_patches = 2 ** level
	patch_size = dimensions // num_patches

	(p1, p2, p3, p4, p5) = generate_patches(patch_size)

	b = 0

	for l in range(level):
		b += (4 ** l)

	b *= 5

	print(b)

	for i in range(num_patches):
		for j in range(num_patches):
			x = patch_size * i
			y = patch_size * j

			f = features[b:b + 5]

			image[x:x + patch_size, y:y + patch_size] += \
				p1 * f[0] + \
				p2 * f[1] + \
				p3 * f[2] + \
				p4 * f[3] + \
				p5 * f[4]

			b += 5

	return image


def generate_patches(s):
	patch1 = np.zeros((s, s))
	patch2 = np.zeros((s, s))
	patch3 = np.zeros((s, s))
	patch4 = np.zeros((s, s))
	patch5 = np.zeros((s, s))

	s2 = s // 2
	s3 = s // 3

	patch1[:s2, :] = 1
	patch1[s2:, :] = -1

	patch2[:, :s2] = 1
	patch2[:, s2:] = -1

	patch3[:s3, :] = 1
	patch3[s3:s3 * 2, :] = -1
	patch3[s3 * 2:, :] = 1

	patch4[:, s3] = 1
	patch4[:, s3:s3 * 2] = -1
	patch4[:, s3 * 2:] = 1

	patch5[:s2, :s2] = 1
	patch5[s2:, s2:] = 1

	patch5[s2:, :s2] = -1
	patch5[:s2, s2:] = -1

	return (patch1, patch2, patch3, patch4, patch5)
