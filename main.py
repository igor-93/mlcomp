from load_data import load, load_big_images
from visualize import vis
from preprocess2_data import preprocess
from preprocess2_data import preprocess_image
from classify_data import train

import matplotlib.pyplot as plt


def main():
	levels = 3

	X, Y = load(3000, lambda x: preprocess_image(x, 0, levels))

	train(X, Y)

	# for i in range(2,4):
	# 	X_processed = preprocess(X, i)
	# 	learn(X_processed,Y)
	

main()
