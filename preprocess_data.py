import numpy as np
from skimage.transform import pyramid_gaussian

def preprocess(X, levels  = 5):
	w = len(X)
	h = levels * 5
	ft_matrix = np.zeros((w,h))
	for idx,x in enumerate(X):
		ft = []
		pyramid = tuple(pyramid_gaussian(x))
		for idx_img,img in enumerate(pyramid):
			ft.append(extract_features(img))
		ft_matrix[idx] = np.array(ft).flatten
	return ft_matrix


def extract_features(img):
	
	
	w,h = img.shape
	
	ft = np.sum(img[:w//2,:]) - np.sum(img[w//2:,:])
	ft.append(np.sum(img[:,:h//2]) - np.sum(img[:,h//2:]))
	
	sl =  np.sum(img[:w//3,:]) - np.sum(img[w//3:w//3*2,:]) + np.sum(img[w//3*2:,:])
	ft.append(sl)
	
	sl = np.sum(img[:,:h//3]) - np.sum(img[:,h//3:h//3*2]) + np.sum(img[:,h//3*2:])
	ft.append(sl)

	sl = np.sum(img[:w//2,:h//2]) + np.sum(img[w//2:,h//2:])
	sl -= np.sum(img[w//2:,:h//2]) + np.sum(img[:w//2,h//2:])

	ft.append(sl)
	
	return np.array(ft)
 	

