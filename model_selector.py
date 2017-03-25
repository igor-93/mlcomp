import numpy as np
from sklearn.model_selection import cross_val_score


def selectClassifier(X, y, classifiers):
    for classifier in classifiers:
        scores = cross_val_score(classifier, X, y)

        classifierName = classifier.__class__.__name__

        print('{}: {}'.format(classifierName, np.sum(scores) / len(scores)))

        # for train, test in loo.split(X, y):
        #     classifier.fit(X, y)
