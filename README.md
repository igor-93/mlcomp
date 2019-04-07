# EESTech Challenge 2017

Face classification & face detection problem solved with random forests.

## How Do you run the Code
\<python-dist\> main.py
## What are the Parameters
* levels -> number of levels we go down in the image pyramid
* bins -> we are concatenating a histogram of the hue of the analyzed picture with the specified number of bins
* number of randomized, shift-based augmentations for positive samples

## What Features did we use
We used two kind of features:
* RGB	
* Hue

We compute a pyramid of features according to the assignment statement. Based on the 5 masks we compute a 
'feature pyramid' with three levels for each color channel of the image.
Additionally, we calculate a n-bin-histogram over the hue-values of the image. This is done because the hue values are distinctive for the faces. 

##How did we test our Predictions 
We used 5-fold cross validation

## How did we improve the Training Data
We augmented our data by adding randomly padded and shifted versions of face-positive images. This way, we improve the robustness of our classifier. 
In fact, this step is crucial regarding our face-detection algorithm: 
Because it is based on k-means, selected tiles tend to contain not only the face but also some significant amount of background pixels.

## Lost in the Code?
* load_data.py helper function for streaming and pre-processing the data
* main.py runs the main pipeline
* detector.py finds bounding boxes for the faces in the big images
* classify_data.py contains the classifier definition

## General Idea of the Pipeline
* Load the data as a stream.
* Data is augmented depending on its label. (c.f. above)
* Train Classifier: Random Forest with the following parameters
    * n_estimators = 1000
    * class_weight = 'balanced'
* Classify Images
* Detection of Bounding Boxes: 
    * Find approximate Face Positions with K-means: We filter the RGB image for the range of skin colour and run k-means on the data points obtained this way.
    * Put growing bounding boxes around centroids and compute the score of the box containing a face
    * Select the best bounding box size based on the maximum score
    * Discard overlapping and redundant boxes

## What libraries did we use?
We used the machine-learning and scientific computing libraries of python, namely:
* numpy
* scipy
* scikit-learn
