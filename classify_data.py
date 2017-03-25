
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from preprocess_data import preprocess_image
from load_data import test_green
import matplotlib.pyplot as plt


n_estimators = 1000
classifier = RandomForestClassifier(n_estimators=n_estimators,class_weight = "balanced")
box_sizes = #[200] [32,64]
min_certainty = 0.5
slide_advance = 25
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
		sub_box = []
		w,h,channels = im.shape
		box_sizes = calculate_box_sizes(w,h)
		for s in box_sizes:
				x_range = range( (w - s) // slide_advance ) 
				y_range = range( (h - s) // slide_advance )
				print(len(x_range) * len(y_range),w,h)
				for x in x_range:
					for y in y_range :	
						# Hue in the patch
						x_min = x*slide_advance
						y_min = y*slide_advance
						patch = im[x_min:x_min+s,y_min:y_min+s]

						
						if(test_green(patch)):
							continue
						X = preprocess_image(patch[:,:,2],min_level,max_level)
						prob = predict(X.reshape(1,-1))					
							
						if prob >= min_certainty:
							print(prob,x_min,y_min)
							#plt.imshow(patch)
							#plt.show()
							sub_box.append([x_min,y_min,s,s])
		print(len(sub_box))
		boxes.append(sub_box)
	return boxes

