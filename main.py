from load_data import load
from visualize import vis
from preprocess2_data import preprocess

import matplotlib.pyplot as plt


def main():
    X, Y = load(40)

    for x in X[20:40]:
        plt.imshow(x)
        plt.show()


    X = preprocess(X, num_levels=3)

    vis(X,Y)

main()
