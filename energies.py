import numpy as np
import scipy.sparse as sps
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm 

from scipy.optimize import nnls

IMPLEMENTED_ENERGIES = ['conic', 'cluster']


class EnergyClass(object):
    def __init__(self, X, p=2):
        self.X = X
        self.k = None 
        self.p = p
        assert (self.p is None) or (self.p > 0)
        self.d, self.n = X.shape
        self.energy = None
        self.indices = []
        self.dists = np.ones(self.n)
        self.energy_values = []
        if sps.issparse(X):
            self.Xfro_norm2 = sps.linalg.norm(self.X, ord='fro')**2.
        else:
            self.Xfro_norm2 = np.linalg.norm(self.X, ord='fro')**2.

    def set_k(self, k):
        assert type(k) == int 
        assert k >= 0
        self.k = k
        return 

    def compute_energy(self):
        if self.p is not None:
            self.energy = np.linalg.norm(self.dists, ord=self.p) 
        else:
            self.energy = np.max(self.dists)
        return 
        
    def add(self, i):
        self.indices.append(i)
        return
    
    def swap(self, t, i):
        return

    def init_set(self, inds):
        assert self.k is None 
        self.set_k(len(inds))
        for i in inds:
            self.add(i)
        return

    def get_search_distances(self, candidates, idx_to_swap=None):
        return 
    
    def compute_search_values(self, candidates=None, idx_to_swap=None):
        if candidates is None:
            candidates = np.delete(np.arange(self.n), self.indices)

        if idx_to_swap is not None:
            assert (idx_to_swap < len(self.indices))  and (idx_to_swap >= 0)
        
        search_dists = self.get_search_distances(candidates, idx_to_swap=idx_to_swap)
        
        all_energy_vals = np.ones(self.n)*self.energy
        if self.p is not None:
            all_energy_vals[candidates] = np.array([np.linalg.norm(s_dist, ord=self.p) for s_dist in search_dists])
        else:
            all_energy_vals[candidates] = np.array([np.max(s_dist) for s_dist in search_dists])

        return all_energy_vals
    
    


class ConicHullEnergy(EnergyClass):
    def __init__(self, X, p=2, n_jobs=4, verbose=False):
        super().__init__(X, p=p)
        assert self.X.min() >= -1e-13 # ensure non-negativity
        self.sparse_flag = sps.issparse(self.X)

        # Compute Euclidean norms raised to the pth power of each row
        if self.sparse_flag:
            self.dists = sps.linalg.norm(self.X, ord=2, axis=0).flatten()
        else:
            self.dists = np.linalg.norm(self.X, ord=2, axis=0).flatten()
        
        self.compute_energy()
        self.use_previous = True
        self.G_diag = self.dists**2.
        self.verbose = verbose
        self.n_jobs = n_jobs
        

    def set_k(self, k):
        super().set_k(k)
        self.W = np.zeros((self.k, self.d))
        self.H = np.zeros((self.k, self.n))
        self.G_S = np.zeros((self.k, self.n)) 
        return 
    
        
    def add(self, i):
        if self.k is None:
            raise NotImplementedError("Iterative allocation of memory for ConicHullEnergy objects not yet implemented. Must set desired k with ConicHullEnergy.set_k(k)")
        if self.sparse_flag:
            self.W[len(self.indices),:] = self.X[:,i].todense().A1.flatten()
        else:
            self.W[len(self.indices),:] = self.X[:,i].flatten()
        self.G_S[len(self.indices),:] = self.X.T @ self.X[:,i].flatten()
        self.indices.append(i)
        dists, H = self.nnls_OGM_gram(returnH=True)
        self.H[:len(self.indices),:] = H 
        self.dists = dists 
        self.compute_energy()
        self.energy_values.append(self.energy)
        return

    def swap(self, t, i):
        assert (t < len(self.indices))  and (t >= 0)
        self.indices[t] = i 
        self.G_S[t,:] = self.X.T @ self.X[:,i].flatten()
        dists, H = self.nnls_OGM_gram(returnH=True) 
        self.H = H   # assuming swap is only done with len(self.indices) = self.k
        self.dists = dists 
        self.compute_energy()
        self.energy_values.append(self.energy)
    
    def get_search_distances(self, candidates, idx_to_swap=None):
        if self.verbose:
            iterator = tqdm(candidates, total=len(candidates))
            iterator.set_description(f"Computing conic hull search values... len(self.indices) = {len(self.indices)}")
        else:
            iterator = candidates
        
        if self.n_jobs is not None:
            with parallel_backend("loky", inner_max_num_threads=1):
                outs = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.nnls_OGM_gram)(search_ind=c, idx_to_swap=idx_to_swap, returnH=False) for c in iterator)
                search_dists = [out[0] for out in outs]
        else:
            search_dists = [self.nnls_OGM_gram(search_ind=c, idx_to_swap=idx_to_swap, returnH=False)[0] for c in iterator]
        
        
        return search_dists 
    

    def nnls_OGM_gram(self, search_ind=None, idx_to_swap=None, delta=1e-3, maxiter=500, lam=1.0, returnH=True, verbose=False, term_cond=1):
        """
        Non-negative Least Squares, Optimal Gradient Method, using the Gram matrix.
        """
        if term_cond == 1:
            assert self.X is not None 
        
        H0 = None 
        if self.use_previous:
            # use previous only if (1) NOT performing swap move and (2) doing a search move with (3) already having computed H previously
            if (idx_to_swap is None) and (search_ind is not None) and (len(self.indices) > 1):
                H0 = np.zeros((len(self.indices)+1, self.H.shape[1]))
                H0[:-1,:] = self.H[:len(self.indices),:].copy()
        
        S_ind_all = self.indices[:]
        if idx_to_swap is not None:
            assert search_ind is not None
            S_ind_all[idx_to_swap] = search_ind 
        else:
            if search_ind is not None:
                S_ind_all = S_ind_all + [search_ind]
        
        
        if len(self.indices) == 0: # case of first iteration
            G_S = self.X[:, S_ind_all].T @ self.X
        else:
            if idx_to_swap is not None: # swap move should only happen when self.G_S.shape[0] == self.k
                assert self.G_S.shape[0] == self.k
                G_S = np.array(self.G_S, copy=True) 
                G_S[idx_to_swap, :] = self.X.T @ self.X[:,search_ind].flatten()
            else:
                if search_ind is None: # just evaluating the projection with the current indices
                    G_S =  np.array(self.G_S[:len(S_ind_all),:], copy=True) 
                else: # this is a build_phase search case, so include the search_ind now in G_S
                    G_S =  np.array(self.G_S[:len(S_ind_all),:], copy=True)
                    G_S[-1,:] = (self.X.T @ self.X[:, search_ind]).flatten()
        
        G_SS = G_S[:,S_ind_all]

        L = np.linalg.norm(G_SS, 2)
        
        if H0 is None:
            H = np.maximum(0.0, np.linalg.pinv(G_SS)@ G_S)
        else:
            H = H0.copy()
        Z = H.copy()

        i = 0
        continue_flag = True
        if verbose:
            EPS = []
        while i <= maxiter and continue_flag:
            Hp = H.copy()
            H = np.maximum(0.0, Z - (G_SS @ Z - G_S)/L)
            lam_ = 0.5*(1. + np.sqrt(1. + 4.*lam**2.))  
            beta = (lam - 1.0)/lam_ 
            Z = H + beta*(H - Hp)

            i += 1
            lam = lam_
            

            if term_cond == 0:
                if i == 1:
                    eps0 = np.linalg.norm(H - Hp, 'fro')
                    eps = eps0
                else:
                    eps = np.linalg.norm(H-Hp, 'fro')
                continue_flag = eps >= delta*eps0
            elif term_cond == 1:
                gradient_proj = G_SS @ H - G_S
                mask = H <= 1e-8
                gradient_proj[mask] = np.minimum(0.0, gradient_proj[mask])
                eps = np.linalg.norm(gradient_proj, ord='fro')
                if i == 1:
                    Mat = H@(H.T @ self.X[:,S_ind_all].T - self.X.T) # don't need to mask out,because we know that W = X[S_ind_all,:] is non-negative
                    eps0 = np.sqrt(eps**2. + np.linalg.norm(Mat, ord='fro')**2.) 
                else:
                    
                    if eps < delta*eps0:
                        if i <= 10:
                            delta *= 0.1
                        else:
                            continue_flag = False 
            else:
                raise ValueError(f"term_cond = {term_cond} not recognized...")
            
            if verbose:
                EPS.append(eps)
        
        dist_vals = self.G_diag - 2.*(G_S * H).sum(axis=0) + ((G_SS @ H) * H).sum(axis=0)
        dist_vals[dist_vals < 0] = 0.0
        dist_vals = np.sqrt(dist_vals / self.Xfro_norm2)    # we consider Euclidean distances and divide by Fro norm of X to 
                                                            # keep values from being too large with high-dimensional datasets
        if verbose:
            return {"dist_vals":dist_vals, "iters":i, "eps":eps, "eps0":eps0, "EPS":EPS }, H

        if returnH:
            return dist_vals, H 
        
        return dist_vals, None





######################################
############# NEED TO BE UPDATED #######
#####################################


FLOAT_THRESHOLD = 20000

class ClusteringEnergy(EnergyClass):
    """
    Not finished with implementing. Need to use whole matrix D (n x n) in search methods since we will be searching over this anyway...
    """
    def __init__(self, X, p=2):
        super().__init__(X, p=p)
        self.d, self.n = self.X.shape
        if self.n >= FLOAT_THRESHOLD:
            self.X = self.X.astype(np.float32)
        self.x_squared_norms = np.linalg.norm(self.X, axis=0)**2.
        self.dists = np.ones(self.n) # initial energy for adaptive sampling should give equal weight to every point
        self.compute_energy()
    
    def set_k(self, k):
        super().set_k(k)
        self.D = np.zeros((self.k, self.n), dtype=self.X.dtype)
        
    def add(self, i):
        self.D[len(self.indices),:] = self.compute_distances(i)
        if len(self.indices) > 0:
            np.minimum(self.D[len(self.indices),:], self.dists, out=self.dists)
        else:
            self.dists = self.D[0,:].copy()
        self.indices.append(i)
        self.compute_energy()
        self.energy_values.append(self.energy)
        return 
    
    def swap(self, t, i):
        assert (t < len(self.indices))  and (t >= 0)
        if self.D is None:
            self.D = self.compute_distances(self.indices)
        self.indices[t] = i 
        self.D[t,:] = self.compute_distances(i)
        self.dists = np.min(self.D, axis=0) 
        self.compute_energy()
        self.energy_values.append(self.energy)
    
    def compute_distances(self, inds):
        if type(inds) in [int, np.int8, np.int16, np.int32, np.int64]:
            dists = euclidean_distances(self.X[:,inds].reshape(1,-1), self.X.T, Y_norm_squared=self.x_squared_norms, \
                                       squared=False).flatten()
        elif type(inds) == list:
            dists = euclidean_distances(self.X[:,inds].T, self.X.T, Y_norm_squared=self.x_squared_norms, \
                                         X_norm_squared=self.x_squared_norms[inds], squared=False)
        else:
            raise ValueError(f"inds must be of type `int` or `list`")
        
        return dists 
    
    def get_search_distances(self, candidates, idx_to_swap=None):
        
        if idx_to_swap is None:
            search_dists = [np.minimum(self.dists, self.compute_distances(c)) for c in candidates]
        else:
            assert (0 <= idx_to_swap) and (idx_to_swap < len(self.indices)) 
            assert self.k == len(self.indices)
            dists_wo_st = np.min(np.vstack((self.D[:idx_to_swap,:], self.D[idx_to_swap+1:,:])), axis=0) 
            search_dists = [np.minimum(dists_wo_st, self.compute_distances(c)) for c in candidates]
        return search_dists



class LowRankEnergy(EnergyClass):
    """
    NOT IMPLEMENTED CURRENTLY
    """
    def __init__(self, X, p=2):
        super().__init__(X, p=p)
        self.F = np.zeros((self.n, self.k))
        self.dists = np.linalg.norm(self.X, ord=2, axis=0).flatten()
        self.energy = self.dists.sum()
        
    def add(self, i):
        if self.kernel is None:
            g = self.X.T @ self.X[:,i].flatten()
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

    


