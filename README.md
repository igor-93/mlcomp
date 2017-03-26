EESTech Challenge 2017

# How Do you run the Code
\<python-dist\> main.py
# What are the Parameters
levels -> number of levels we go down in the image pyramid
# What Features did we use
We did use two kind of features:
* RGB	

* Hue
We do a histogram with n bins over the huevalues of the image. This is done because the hue values are distinctive for the faces. 
#How did we test our Predictions 
We did use 5-fold cross validation
# How did we improve the Training Data
We added transformed face images to the training data. We translate them randomly in order to mirror the reality of face-recognition better.
# Classes
*main runs the main pipeline
*detector finds bounding boxes for the faces in the big images
# General Idea of the Pipeline
*Load the Data
Data is augmented in the described way and loaded
*Train Classifier
We train a Random Forest with the following parameters
** n_estimators = 1000
** class_weight = 'balanced'
*Classify Images
*Detection of Bounding Boxes
This is separated in two parts
*Find approximate Face Positions with K-means
We filter the RGB image for the range of skin colour and run k-means on the data points obtained this way.
*Put Bounding Box around Centroid
We start enlarging boxes around the centroids and take the box with the largest likelihood
*Maximum Suppression
Do the maximum suppression for the bounding boxes to obtain the best positioning for the faces


