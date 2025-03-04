import numpy as np
import scipy.sparse as sps
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from joblib import Parallel, delayed
from tqdm import tqdm 

from scipy.optimize import nnls

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
        
        cand_dists = euclidean_distances(self.X[candidates,:], self.X, Y_norm_squared=self.x_squared_norms, squared=True)
        np.minimum(self.dists, cand_dists, out=cand_dists)
        candidate_energy_vals = cand_dists.sum(axis=1).flatten()
        return candidate_energy_vals

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


CHUNK_SIZE = 750

'''
TODO: Make nnls_ogm_gram a member function so it can have access to self.X for termination condition in conical hull setting.
'''
class ConvexHullEnergy(Energy):
    def __init__(self, X, k, n_jobs=4, compute_gram=True, hull=False):
        super().__init__(X, k)
        assert self.X.min() >= -1e-13 # ensure non-negativity
        self.sparse_flag = sps.issparse(self.X)
        self.W = np.zeros((self.k, self.d))
        self.H = np.zeros((self.k, self.n))
        if self.sparse_flag:
            self.dists = sps.linalg.norm(self.X, ord=2, axis=1).flatten()**2.
        else:
            self.dists = (self.X * self.X).sum(axis=1).flatten()
        self.energy = self.dists.sum()
        self.p_init = self.dists / self.energy    
        self.n_jobs = n_jobs
        self.unselected_inds = np.arange(self.n)
        self.chunk_size = min(CHUNK_SIZE, self.n / 2)
        self.chunk_inds = np.array_split(np.arange(self.n), self.n//self.chunk_size)
        self.hull = hull
        self.use_previous = True
        self.G_diag = (X * X).sum(axis=1)
        self.G_S = np.empty((self.k, self.n)) 
        
        
    def add(self, i):
        if self.sparse_flag:
            self.W[self.k_sel,:] = self.X[i,:].todense().A1
        else:
            self.W[self.k_sel,:] = self.X[i,:]
        self.indices.append(i)
        self.G_S[self.k_sel,:] = self.X @ self.X[i,:]
        self.k_sel += 1
        self.unselected_inds = np.delete(np.arange(self.n), self.indices)
        energy, dists, H = self.compute_projection(self.indices, returnH=True)
        self.H = H 
        self.dists = dists 
        self.energy = energy 
        self.energy_values.append(self.energy)

        assert self.dists.size == self.n
        assert self.H.shape[1] == self.n
        return
    
    def init_set(self, inds): # done so that we can track the energy values throughout all choices.
        assert len(inds) == self.k
        for i in inds:
            self.add(i)
        return
    
    def nnls_OGM_gram(self, S_ind, delta=1e-3, maxiter=500, lam=1.0, H0=None, returnH=True, verbose=False, term_cond=1):
        """
        G_S = |S_ind| x n  numpy array Gram submatrix
        """
        if term_cond == 1:
            assert self.X is not None 

        if self.k_sel == 0:
            G_S = self.X[S_ind,:] @ self.X.T
        else:
            G_S = self.G_S[:len(S_ind),:] # make copy since will be using often
        G_SS = G_S[:,S_ind]

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
                    Mat = H@(H.T @ self.X[S_ind,:] - self.X) # don't need to mask out,because we know that W = X[S_ind,:] is non-negative
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
        
        energy_vals = self.G_diag - 2.*(G_S * H).sum(axis=0) + ((G_SS @ H) * H).sum(axis=0)
        energy_vals[energy_vals < 0] = 0.0
        if verbose:
            return H, {"energy_vals":energy_vals, "iters":i, "eps":eps, "eps0":eps0, "EPS":EPS }

        if returnH:
            return H, energy_vals
        return None, energy_vals

    def compute_projection(self, inds, returnH=False):
        k_ = 0
        H0 = None 
        if self.use_previous:
            if self.k_sel > 1:
                H0 = np.zeros((self.H.shape[0]+1, self.H.shape[1]))
                H0[:-1,:] = self.H.copy()
        H, energy_vals = self.nnls_OGM_gram(inds, returnH=returnH, H0=H0)
        dists = energy_vals/self.Xfro_norm2 
        return dists.sum(), dists, H 
        


    def look_ahead(self, candidates=None, verbose=False):
        
        if candidates is None:
            candidates = self.unselected_inds

        if verbose:
            iterator = tqdm(candidates, total=len(candidates))
            iterator.set_description("Computing look-ahead values...")
        else:
            iterator = candidates 

        outs = Parallel(n_jobs=self.n_jobs)(
            delayed(self.compute_projection)(self.indices + [c]) for c in iterator)
        
        candidate_energy_vals = np.array([out[0] for out in outs])
        return candidate_energy_vals

    def swap_move(self, method, j_adap=None, verbose=True):
        if self.G is None: # swap moves we have implemented for gram matrix-enabled only
            self.G = self.X.T @ self.X
            self.G_diag = np.diagonal(self.G)
            if self.sparse_flag:
                self.G = self.G.todense()
        
        continue_swap = True
        
        if method == "greedyla":
            if verbose:
                iterator = tqdm(enumerate(self.unselected_inds), total=len(self.unselected_inds))
                iterator.set_description("Computing greedyla swap move...")
            else:
                iterator = enumerate(self.unselected_inds)

            def get_row(i, idx_i):
                row = []
                indices_ij = np.copy(self.indices)
                for j, idx_j in enumerate(self.indices):
                    indices_ij[j] = idx_i 
                    energy, _, _ = self.compute_projection(indices_ij)# compute what the swap's energy would be
                    row.append(energy)
                    indices_ij[j] = idx_j 
                return row
            
            C = Parallel(n_jobs=self.n_jobs)(
                delayed(get_row)(i, idx_i) for i, idx_i in iterator)
            C = np.array(C)
            
            # identify best possible current swap
            js = np.argmin(C, axis=1, keepdims=True)
            best_energy_vals = np.take_along_axis(C, js, axis=1)
            i = np.argmin(best_energy_vals)
            val = best_energy_vals[i]

            if val < self.energy: # if have lower resulting energy, then we will swap
                idx_i = self.unselected_inds[i]
                j = js.flatten()[i]
                idx_j = self.indices[j]

                # swap the indices
                self.indices[j] = idx_i
                self.unselected_inds[i] = idx_j

                # we have to recompute the projection to get the H matrix, since we didn't record it for every possible candidate to save on space
                energy, dists, H = self.compute_projection(self.indices, returnH=True)
                self.H = H 
                self.dists = dists 
                self.energy = energy
                self.energy_values.append(self.energy)
        
            else:
                continue_swap = False
                

            
        elif method[:2] == "p-" or method == "greedy":
            assert j_adap is not None 
            
            indices_wo_jadap = self.indices[:j_adap] + self.indices[j_adap+1:]
            energy_wo_jadap, dists_wo_jadap, _ = self.compute_projection(indices_wo_jadap)
            assert dists_wo_jadap.size == self.n
            
            if method == "greedy": # arbitrarily choose a point that is the max distance
                max_dist = np.max(dists_wo_jadap)
                max_dist_inds = np.where(dists_wo_jadap == max_dist)[0]
                idx_j_new = np.random.choice(max_dist_inds)
                if idx_j_new == self.indices[j_adap]:
                    continue_swap = False   # this flag will be used differently for p-inf than greedy look-ahead. will be used to count number of successive steps of no changes when p = \infty

            else:
                p = float(method[2:])
                q_probs = np.power(dists_wo_jadap, p/2.0)  # need to divide p by 2 because the dists are actually squared distances. 
                if np.isclose(q_probs.sum(), 0.0):
                    idx_j_new = self.indices[j_adap] # we've gotten the lowest possible energy, so stop swapping
                    continue_swap = False 
                else:
                    q_probs /= q_probs.sum()
                    idx_j_new = np.random.choice(self.n, p=q_probs)
            
            # update object and corresponding energy
            self.indices[j_adap] = idx_j_new
            self.unselected_inds = np.delete(np.arange(self.n), self.indices)
            energy, dists, H = self.compute_projection(self.indices, returnH=True)
            self.H = H
            self.dists = dists 
            self.energy = energy 
            self.energy_values.append(self.energy)

        else:
            raise ValueError(f"swap_method = {method} not recognized for swap moves...")
        

        # return whether or not to continue swapping (based on if we've found a local minimizer in greedy setting)
        return continue_swap 