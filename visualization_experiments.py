from load_data import load
from visualize import vis
from preprocess2_data import preprocess

from visualize_patches import features2image
from visualize_patches import features2image2

import matplotlib.pyplot as plt


def main():
    I, Y = load(40)

    I = I[20:40]
    X = preprocess(I, num_levels=4)

    for x, i in zip(X, I):
        plt.imshow(i)
        plt.show()
        plt.imshow(features2image2(x, level=4))
        plt.show()


    vis(X,Y)

main()
