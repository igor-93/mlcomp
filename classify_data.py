
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score


#classifier = SVC(gamma=2, C=1)

n_estimators = 150
classifier = RandomForestClassifier(n_estimators=n_estimators)

def learn(X, y):
	
#	classifier.fit(X, y)
#	print(classifier.score(X,y))
#	print(classifier.oob_score_)
	scores = cross_val_score(classifier, X, y, cv=5, scoring='neg_log_loss')
	print(scores.mean(),scores.std())
	#classifier1.fit(X, first_lvl_targets)
