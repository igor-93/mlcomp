import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from preprocess_data import preprocess_image
from load_data import *

n_estimators = 1000
classifier = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",n_jobs = -1)
min_certainty = 0.5
max_level = 3
min_level = 0


def evaluate_performance(X, y):
	scores = cross_val_score(classifier, X, y, cv=5, scoring='neg_log_loss')
	print(scores.mean(), scores.std())


def evaluate_performance_FPFN(X, y):
	result = classifier.predict(X)

	TP = 0
	TN = 0
	FN = 0
	FP = 0

	for a, b in zip(result, y):
		if a != b:
			if a == 0:
				FN += 1
			else:
				FP += 1
		else:
			if a == 0:
				TN += 1
			else:
				TP += 1

	print("FP: {}; FN: {}; TP: {}; TN: {}".format(FP, FN, TP, TN))


def train(X, y):
	classifier.fit(X, y)


def predict(X):
	return classifier.predict_proba(X)[:, 1]
