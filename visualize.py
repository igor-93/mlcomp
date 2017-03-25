from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.decomposition import PCA 
import numpy as np
import matplotlib.pyplot as plt


def vis(X,Y):
	L = PCA(n_components =  2)
	#L_vis = PCA(n_components = 100).fit(X)
	X = L.fit_transform(X)
	# eig = L_vis.explained_variance_ratio_
	#print(eig)
	fig, (low_dim,eigenvalues) = plt.subplots(2,1)
	low_dim.scatter(X[:,0],X[:,1],c = Y)
	#eigenvalues.plot(range(len(eig)),eig)
	plt.show()
	

