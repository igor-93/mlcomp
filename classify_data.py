
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM


n_estimators = 1000
classifier = RandomForestClassifier(n_estimators=n_estimators,class_weight = "balanced")


def evaluate_performance(X, y):
	scores = cross_val_score(classifier, X, y, cv=5, scoring='neg_log_loss')
	print(scores.mean(),scores.std())

def train(X,y):
	classifier.fit(X,y)

def predict(Y):
	return classifier.predict_proba(Y)
