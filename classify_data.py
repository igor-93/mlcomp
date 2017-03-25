
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from preprocess_data import preprocess_image


n_estimators = 1000
classifier = RandomForestClassifier(n_estimators=n_estimators,class_weight = "balanced")
box_sizes = [32,64]
min_certainty = 0.75
slide_advance = 10
level = 3

def evaluate_performance(X, y):
	scores = cross_val_score(classifier, X, y, cv=5, scoring='neg_log_loss')
	print(scores.mean(),scores.std())

def train(X,y):
	classifier.fit(X,y)

def predict(X):
	return classifier.predict_proba(X)




def extract_boxes(images):
	boxes = []
	for im in images:
		sub_box = []
		w,h = im.shape
		for s in box_sizes:
				for x in range( (w - s) // slide_advance ) :
					for y in range( (h - s) // slide_advance ) :	
						# Hue in the patch
						patch = im[x:x+s,y:y+s,0]
						if(hue_test(patch)):
							continue
						patch = preprocess_image(patch,level)
						prob = predict(patch)							
						if prob >= min_certainty:
							sub_box.append(x,y,x+s,y+s)
		boxes.append(sub_box)
	return boxes

