from load_data import load, load_big_images
from visualize import vis
from preprocess_data import preprocess_image
from classify_data import *
from write_solution import *

import matplotlib.pyplot as plt


def main():
	levels = 3

	X, Y = load(1000, lambda x: preprocess_image(x, 0, levels))
	
	
	train(X, Y)
	evaluate_performance(X,Y)
	
	
	big_images = load_big_images()
	boxes = extract_boxes(big_images)
	write_output_detection(big_images,boxes)
	

main()
