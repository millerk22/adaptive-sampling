import numpy as np
import scipy.sparse as sps
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from joblib import Parallel, delayed
from tqdm import tqdm 

from scipy.optimize import nnls
from nnls_FPGM import nnls_FPGM

class Energy(object):
    def __init__(self, X, k):
        self.X = X
        self.k = k
        self.n, self.d = X.shape
        self.energy = None
        self.k_sel = 0
        self.indices = []
        self.dists = np.ones(self.n)
        self.energy_values = []
        if sps.issparse(X):
            self.Xfro_norm2 = sps.linalg.norm(self.X, ord='fro')**2.
        else:
            self.Xfro_norm2 = np.linalg.norm(self.X, ord='fro')**2.
        
    def add(self, i):
        return

    def look_ahead(self, candidates):
        return

class KmeansEnergy(Energy):
    def __init__(self, X, k, knn=None):
        super().__init__(X, k)
        self.x_squared_norms = (X * X).sum(axis=1)
        self.p_init = np.ones(self.n) / float(self.n)
        self.knn = knn
        
    def add(self, i):
        i_dists  = euclidean_distances(
                        self.X[i,:].reshape(1,-1), self.X, Y_norm_squared=self.x_squared_norms, squared=True
                        ).flatten()
        if self.knn:
            knn_dist_i = np.sort(i_dists)[self.knn+1]
            mask = i_dists > knn_dist_i
            i_dists /= knn_dist_i
            i_dists[mask] += 9.
            
            
        if self.k_sel > 0:
            np.minimum(self.dists, i_dists, out=self.dists)
        else:
            self.dists = i_dists
        
        self.indices.append(i)
        self.energy = self.dists.sum()
        self.k_sel += 1
        self.energy_values.append(self.energy)
        return 
    
    def init_set(self, inds):
        assert len(inds) == self.k
        for i in inds:
            self.add(i)
        return

    def look_ahead(self, candidates=None):
        if candidates is None:
            candidates = np.delete(np.arange(self.n), self.indices)
        cand_la_results = {c : {} for c in candidates}
        cand_dists = euclidean_distances(self.X[candidates,:], self.X, Y_norm_squared=self.x_squared_norms, squared=True)
        np.minimum(self.dists, cand_dists, out=cand_dists)
        for i, c in enumerate(candidates):
            cand_la_results[c]['dists'] = cand_dists[i,:]
            cand_la_results[c]['energy'] = cand_la_results[c]['dists'].sum()
        return cand_la_results

    def update_from_look_ahead(self, c, choice_dict):
        self.dists = choice_dict['dists']
        self.energy = choice_dict['energy']
        self.energy_values.append(self.energy)
        self.indices.append(c)
        self.k_sel += 1
        self.unselected_inds = np.setdiff1d(self.unselected_inds, [c])
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
        if self.k_sel > 0:
            g -= self.F[:,:self.k_sel] @ self.F[i, :self.k_sel]
        if np.isclose(g[i], 0) or g[i] < 0:
            print(f"Iter = {self.k_sel+1}, g[i] approx 0, exiting...")
            self.k_sel += 1
            return 
        self.F[:,self.k_sel] = g / np.sqrt(g[i])
        self.dists -= self.F[:,self.k_sel]**2.
        self.energy = self.dists.sum()
        self.indices.append(i)
        self.k_sel += 1
        self.energy_values.append(self.energy)
        return
    
    def init_set(self, inds):
        assert len(inds) == self.k
        for i in inds:
            self.add(i)
        return

    def look_ahead(self, candidates=None):
        raise NotImplementedError()

    def update_from_look_ahead(self, c, choice_dict):
        raise NotImplementedError() 


CHUNK_SIZE = 500


class ConvexHullEnergy(Energy):
    def __init__(self, X, k, n_jobs=4):
        super().__init__(X, k)
        self.sparse_flag = sps.issparse(self.X)
        self.W = np.zeros((self.k, self.d))
        self.H = np.zeros((self.n, self.k))
        if self.sparse_flag:
            self.dists = sps.linalg.norm(self.X, ord=2, axis=1).flatten()**2.
        else:
            self.dists = (self.X * self.X).sum(axis=1).flatten()
        self.energy = self.dists.sum()
        self.p_init = self.dists / self.energy    
        self.n_jobs = n_jobs
        self.chunk_size = min(CHUNK_SIZE, self.n / 2)
        
        
    def add(self, i):
        if self.sparse_flag:
            self.W[self.k_sel,:] = self.X[i,:].todense().A1
        else:
            self.W[self.k_sel,:] = self.X[i,:]
        self.indices.append(i)
        self.k_sel += 1
        self.unselected_inds = np.delete(np.arange(self.n), self.indices)
        chunk_inds = np.array_split(self.unselected_inds, self.unselected_inds.size//self.chunk_size)
        outs = Parallel(n_jobs=self.n_jobs)(
                    delayed(nnls_FPGM)(self.X[chunk,:].T, self.W[:self.k_sel,:].T) for chunk in chunk_inds
                )
        self.H[self.indices,:self.k_sel] = np.eye(self.k_sel)
        self.H[self.unselected_inds,:self.k_sel] = np.vstack([out[0].T for out in outs])
        self.dists[self.indices] = 0.0
        self.dists[self.unselected_inds] = np.concatenate([out[1]/self.Xfro_norm2 for out in outs])
        self.energy = self.dists.sum()
        self.energy_values.append(self.energy)
        return
    
    def init_set(self, inds): # done so that we can track the energy values throughout all choices.
        assert len(inds) == self.k
        for i in inds:
            self.add(i)
        return

    def look_ahead(self, candidates=None):
        if candidates is None:
            candidates = np.delete(np.arange(self.n), self.indices)
        cand_la_results = {c:{} for c in candidates}
        for c in candidates:
            chunk_inds = np.array_split(self.unselected_inds, self.unselected_inds.size//self.chunk_size)
            outs = Parallel(n_jobs=self.n_jobs)(
                    delayed(nnls_FPGM)(self.X[chunk,:].T, self.W[:self.k_sel,:].T, returnH=False) for chunk in chunk_inds
                )
            dists_c = np.zeros(self.n)
            dists_c[np.setdiff1d(self.unselected_inds, [c])] = np.concatenate([out[1]/self.Xfrom_norm2 for out in outs])
            cand_la_results[c]['dists'] = dists_c
            cand_la_results[c]['energy'] = cand_la_results[c]['dists'].sum()
            
        return cand_la_results

    def update_from_look_ahead(self, c, choice_dict):
        print('Doing look ahead with nnls from scipy...')
        if self.sparse_flag:
            self.W[self.k_sel,:] = self.X[c,:].todense().A1
        else:
            self.W[self.k_sel,:] = self.X[c,:]
        self.indices.append(c)
        self.k_sel += 1
        self.unselected_inds = np.setdiff1d(self.unselected_inds, [c])
        self.dists = choice_dict['dists']
        self.energy = choice_dict['energy']
        self.energy_values.append(self.energy)
        outs = Parallel(n_jobs=self.n_jobs)(
                    delayed(nnls)(self.W[:self.k_sel,:].T, self.X[j,:]) for j in self.unselected_inds
                )
        self.H[self.indices, :self.k_sel] = np.eye(self.k_sel)
        self.H[self.unselected_inds, :self.k_sel] = np.vstack([out[0] for out in outs])
        return 
