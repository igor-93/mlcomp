from load_data import load
from visualize import vis
from preprocess2_data import preprocess
from classify_data import *

import matplotlib.pyplot as plt


def main():
	X, y = load(1000)

	X = preprocess(..)
	classify_data.train(X,y)
	X_test = load_test()
	Y_test = classify_data.predict(X_test)
	write_solution.write_output_classification(Y_test)
	
	big_images= load_data2()
	boxes = extract_boxes(big_images) #big images
	write_solution.write_output_detection(big_images,boxes)
	
	

main()
