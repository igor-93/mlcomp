
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from preprocess_data import preprocess_image
from preprocess_data import preprocess_partial
from load_data import *
import matplotlib.pyplot as plt


n_estimators = 1000
classifier = RandomForestClassifier(n_estimators=n_estimators,class_weight = "balanced")
min_certainty = 0.5
max_level = 3
min_level = 0
def evaluate_performance(X, y):
	scores = cross_val_score(classifier, X, y, cv=5, scoring='neg_log_loss')
	print(scores.mean(),scores.std())

def train(X,y):
	classifier.fit(X,y)

def predict(X):
	return classifier.predict_proba(X)[:,1]

def extract_boxes(images):
	boxes = []

	for im in images:
		
		coords = map_to_bool(im)
		centers = kmeans_img(coords)
		print('KMeans done...')
		#plt.imshow(im)
		#plt.scatter(c[:,0], c[:,1], color='yellow')
		
		#plt.show()
		#data = im[:,:,2]
		w,h, channels = im.shape
		box_sizes = calculate_box_sizes(w,h)
		
		sub_box = []
		for c in centers:
			

			max_prob = 0
			best_box = None 
			for s in box_sizes:

				x = np.max([0,int(c[0]) - s//2])
				x_max = np.min([w,x + s])
				y = np.max([0,int(c[1]) - s//2])
				y_max = np.min([h,y + s])
				patch = im[x:x_max,y:y_max,:]
				if ((x_max -x) != (y_max - y)) or x_max -x == 0 or y_max -y == 0:
					continue
				print('patch shape', patch.shape, c,im.shape,x,x_max,y,y_max,s)

				X = preprocess_image(patch[:,:,2],0,3)
				X = np.hstack([X,hue_histogramm(patch[:,:,0],10)])
				prob = predict(X.reshape(1,-1))
				if prob > max_prob:
					best_box = [x,y,x_max - x,y_max -y]
					max_prob = prob
			if max_prob > min_certainty:
				sub_box.append(best_box)
		boxes.append(sub_box)
	
	return boxes


def calculate_box_sizes(w,h):
	mid = int(w/10.0)
	return [mid -20,mid,mid + 20]
