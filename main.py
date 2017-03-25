from load_data import load
from visualize import vis
from preprocess2_data import preprocess
from preprocess2_data import preprocess_image
from classify_data import learn

import matplotlib.pyplot as plt


def main():
	levels = 3

	X, Y = load(3000, lambda x: preprocess_image(x, 0, levels))

	learn(X, Y)

	# for i in range(2,4):
	# 	X_processed = preprocess(X, i)
	# 	learn(X_processed,Y)
	

main()
