import numpy as np
import numpy as np
from scipy.optimize import nnls
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from tqdm import tqdm 

class Energy(object):
    def __init__(self, X, k):
        self.X = X
        self.k = k
        self.n, self.d = X.shape
        self.energy = None
        self.idx = 0
        self.indices = []
        
    def add(self, i):
        return

class KmeansEnergy(Energy):
    def __init__(self, X, k):
        super().__init__(X, k)
        self.x_squared_norms = (X * X).sum(axis=1)
        self.p_init = np.ones(self.n) / float(self.n)
    
    def add(self, i):
        i_dists  = euclidean_distances(
                        self.X[i,:].reshape(1,-1), self.X, Y_norm_squared=self.x_squared_norms, squared=True
                        ).flatten()
        if self.idx > 0:
            np.minimum(self.dists, i_dists, out=self.dists)
        else:
            self.dists = i_dists
        
        self.indices.append(i)
        self.energy = self.dists.sum()
        return 
    

class LpSubspaceEnergy(Energy):
    def __init__(self, X, k, p=2, kernel=None):
        super().__init__(X, k)
        self.F = np.zeros((self.n, self.k))
        self.kernel = kernel
        
        if self.kernel is None: # just use the usual Euclidean distances
            if p == 2:
                self.dists = (self.X * self.X).sum(axis=1)
            else:
                raise NotImplemented("Not implemented for p != 2")
        else:
            # currently assuming that k(x,x) = 1 for all x 
            self.dists = np.ones(self.n)
            
        self.energy = self.dists.sum()
        self.p_init = self.dists / self.energy   
        
    def add(self, i):
        if self.kernel is None:
            g = self.X @ self.X[i,:]
        else:
            g = self.kernel(self.X[i,:].reshape(1,-1), self.X).flatten()
        if self.idx > 0:
            g -= self.F[:,:self.idx] @ self.F[i, :self.idx]
        if np.isclose(g[i], 0) or g[i] < 0:
            print(f"Iter = {self.idx+1}, g[i] approx 0, exiting...")
            self.idx += 1
            return 
        self.F[:,self.idx] = g / np.sqrt(g[i])
        self.dists -= self.F[:,self.idx]**2.
        self.energy = self.dists.sum()
        self.indices.append(i)
        self.idx += 1
        return


class ConvexHullEnergy(Energy):
    def __init__(self, X, k):
        super().__init__(X, k)
        self.W = np.zeros((self.k, self.d))
        self.H = np.zeros((self.n, self.k))
        self.dists = (self.X * self.X).sum(axis=1).flatten()
        self.energy = self.dists.sum()
        self.p_init = self.dists / self.energy    
        
    def add(self, i):
        self.W[self.idx,:] = self.X[i,:]
        for j in range(self.n):
            self.H[j,:self.idx+1], _ = nnls(self.W[:self.idx+1,:].T, self.X[j,:])
        res = self.X - self.H[:,:self.idx+1] @ self.W[:self.idx+1,:] 
        self.dists = (res * res).sum(axis=1).flatten()
        self.energy = self.dists.sum()
        self.indices.append(i)
        self.idx += 1
        return