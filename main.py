from load_data import load
from visualize import vis
from preprocess2_data import preprocess
from classify_data import learn

import matplotlib.pyplot as plt


def main():
	X, Y = load(2000)
	for i in range(2,4):
		X_processed = preprocess(X, num_levels=i)
		learn(X_processed,Y)
	

main()
