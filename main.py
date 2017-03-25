from load_data import load 
from visualize import vis
from preprocess_data import preprocess


def main():
	X,Y = load(2000)
	X = preprocess(X,levels = 5)
	vis(X,Y)

main()
