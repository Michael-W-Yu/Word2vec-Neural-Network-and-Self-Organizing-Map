import numpy as np

def pca(data):
	#data is a matrix
	#finds pca using svd
	m = data.mean(0)
	D = data.copy()
	for r in range(data.shape[0]):
		D[r] = D[r]-m
	U,S,V = np.linalg.svd(D,full_matrices = False)
	eigenval = S*S/(data.shape[0]-1)
	pd = (V@D.transpose()).transpose()
	return pd, V, eigenval

def init_weight(eigenV, map_sz):
	pca_1 = eigenV[0]
	pca_2 = eigenV[1]
	xs = np.linspace(-1,1,num = map_sz)
	ys = np.linspace(-1,1,num = map_sz)
	wts = np.zeros((map_sz, map_sz, eigenV.shape[0]))
	for i in range(map_sz):
		for j in range(map_sz):
			wts[i][j] = xs[i]*pca_1 + ys[j]*pca_2
	#print(wts)
	return wts
