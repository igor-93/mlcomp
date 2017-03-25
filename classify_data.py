
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


#classifier = SVC(gamma=2, C=1)

n_estimators = 1000
classifier = RandomForestClassifier(n_estimators=n_estimators,class_weight = "balanced")

def learn(X, y):
	
#	classifier.fit(X, y)
#	print(classifier.score(X,y))
#	print(classifier.oob_score_)
	scores = cross_val_score(classifier, X, y, cv=5, scoring='neg_log_loss')
	print(scores.mean(),scores.std())
	#classifier1.fit(X, first_lvl_targets)
