from load_data import load
from visualize import vis
from preprocess2_data import preprocess


def main():
    X, Y = load(1)
    X = preprocess(X, num_levels=2)

    print(X)

    vis(X,Y)

main()
