
import numpy as np



classifier = SVC(gamma=2, C=1)

def learn(X, y):
    classifier.fit(X, y)
