import numpy as np
import scipy.sparse as sps
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from joblib import Parallel, delayed
from tqdm import tqdm 

from scipy.optimize import nnls
from nnls import nnls_FPGM, nnls_OGM, nnls_OGM_gram

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


CHUNK_SIZE = 750


class ConvexHullEnergy(Energy):
    def __init__(self, X, k, n_jobs=4, compute_gram=True):
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
        self.unselected_inds = np.arange(self.n)
        self.chunk_size = min(CHUNK_SIZE, self.n / 2)
        self.chunk_inds = np.array_split(np.arange(self.n), self.n//self.chunk_size)
        self.G = None
        if compute_gram:
            self.G = X @ X.T
            self.G_diag = np.diagonal(self.G)
            if self.sparse_flag:
                self.G = self.G.todense()
        
        
    def add(self, i):
        if self.sparse_flag:
            self.W[self.k_sel,:] = self.X[i,:].todense().A1
        else:
            self.W[self.k_sel,:] = self.X[i,:]
        self.indices.append(i)
        self.k_sel += 1
        self.unselected_inds = np.delete(np.arange(self.n), self.indices)
        # chunk_inds = np.array_split(self.unselected_inds, self.unselected_inds.size//self.chunk_size)
        if self.G is not None:
            outs = Parallel(n_jobs=self.n_jobs)(delayed(nnls_OGM_gram)(self.G[np.ix_(self.indices, np.concatenate((self.indices, chunk)))], np.arange(self.k_sel), G_diag=self.G_diag[np.concatenate((self.indices, chunk))]) for chunk in self.chunk_inds)
            self.H = np.vstack([out[0][:,self.k_sel:].T for out in outs])
            self.dists = np.concatenate([out[1][self.k_sel:]/self.Xfro_norm2 for out in outs])
        else:
            outs = Parallel(n_jobs=self.n_jobs)(
                    delayed(nnls_OGM)(self.X[chunk,:].T, self.W[:self.k_sel,:].T) for chunk in self.chunk_inds)
            self.H = np.vstack([out[0].T for out in outs])
            self.dists = np.concatenate([out[1]/self.Xfro_norm2 for out in outs])
        # self.H[self.indices,:self.k_sel] = np.eye(self.k_sel)
        # self.H[self.unselected_inds,:self.k_sel] = np.vstack([out[0].T for out in outs])
        # self.dists[self.indices] = 0.0
        # self.dists[self.unselected_inds] = np.concatenate([out[1]/self.Xfro_norm2 for out in outs])
        
        assert self.dists.size == self.n
        self.energy = self.dists.sum()
        self.energy_values.append(self.energy)
        return
    
    def init_set(self, inds): # done so that we can track the energy values throughout all choices.
        assert len(inds) == self.k
        for i in inds:
            self.add(i)
        return

    def look_ahead(self, candidates=None, verbose=True):
        if candidates is None:
            candidates = self.unselected_inds
        cand_la_results = {c:{} for c in candidates}
        if verbose:
            iterator = tqdm(candidates, total=len(candidates))
            iterator.set_description("Computing look-ahead values...")
        else:
            iterator = candidates
        for c in iterator:
            # chunk_inds = np.array_split(self.unselected_inds, self.unselected_inds.size//self.chunk_size)
            if self.G is not None:
                outs = Parallel(n_jobs=self.n_jobs)(delayed(nnls_OGM_gram)(self.G[np.ix_(self.indices, np.concatenate((self.indices, chunk)))], np.arange(self.k_sel), G_diag=self.G_diag[np.concatenate((self.indices, chunk))], returnH=False) for chunk in self.chunk_inds)
                dists_c = np.concatenate([out[1][self.k_sel:]/self.Xfro_norm2 for out in outs])
            else:
                outs = Parallel(n_jobs=self.n_jobs)(
                        delayed(nnls_OGM)(self.X[chunk,:].T, self.W[:self.k_sel,:].T, returnH=False) for chunk in self.chunk_inds)
                dists_c = np.concatenate([out[1]/self.Xfro_norm2 for out in outs])
            # dists_c = np.zeros(self.n)
            # dists_c[np.setdiff1d(self.unselected_inds, [c])] = np.concatenate([out[1]/self.Xfro_norm2 for out in outs])
            
            assert dists_c.size == self.n
            cand_la_results[c]['dists'] = dists_c
            cand_la_results[c]['energy'] = cand_la_results[c]['dists'].sum()
            
        return cand_la_results

    def update_from_look_ahead(self, c, choice_dict):
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

        # we have to recompute the projection to get the H matrix, since in the look_ahead function call we didn't record it for every possible candidate to save on space
        # chunk_inds = np.array_split(self.unselected_inds, self.unselected_inds.size//self.chunk_size)
        if self.G is not None:
            outs = Parallel(n_jobs=self.n_jobs)(delayed(nnls_OGM_gram)(self.G[np.ix_(self.indices, np.concatenate((self.indices, chunk)))], np.arange(self.k_sel), G_diag=self.G_diag[np.concatenate((self.indices, chunk))], returnH=True) for chunk in self.chunk_inds)
            self.H = np.vstack([out[0][:,self.k_sel:].T for out in outs])
        else:
            outs = Parallel(n_jobs=self.n_jobs)(
                    delayed(nnls_OGM)(self.X[chunk,:].T, self.W[:self.k_sel,:].T, returnH=True) for chunk in self.chunk_inds)
            self.H = np.vstack([out[0].T for out in outs])
        # self.H[self.indices, :self.k_sel] = np.eye(self.k_sel)
        # self.H[self.unselected_inds, :self.k_sel] = np.vstack([out[0].T for out in outs])
        
        assert self.H.shape[0] == self.n

        return 
    
    def swap_move(self, method, j_adap=None, verbose=True):
        if self.G is None: # swap moves we have implemented for gram matrix-enabled only
            self.G = self.X.T @ self.X
            self.G_diag = np.diagonal(self.G)
            if self.sparse_flag:
                self.G = self.G.todense()
        
        continue_swap = True
        
        if method == "greedyla":
            C = np.zeros((self.n-self.k, self.k))
            if verbose:
                iterator = tqdm(enumerate(self.unselected_inds), total=len(self.unselected_inds))
                iterator.set_description("Computing greedyla swap move...")
            else:
                iterator = enumerate(self.unselected_inds)
            # unselected_inds_ij = np.copy(self.unselected_inds)
            indices_ij = np.copy(self.indices)

            for i, idx_i in iterator:
                for j, idx_j in enumerate(self.indices):
                    # swap the indices
                    indices_ij[j] = idx_i
                    # unselected_inds_ij[i] = idx_j

                    # compute the energy with idx_i and idx_j swapped
                    # chunk_inds = np.array_split(unselected_inds_ij, unselected_inds_ij.size//self.chunk_size)
                    outs = Parallel(n_jobs=self.n_jobs)(delayed(nnls_OGM_gram)(self.G[np.ix_(indices_ij, np.concatenate((indices_ij, chunk)))], np.arange(self.k), G_diag=self.G_diag[np.concatenate((indices_ij, chunk))], returnH=False) for chunk in self.chunk_inds)      
                    C[i, j] = np.concatenate([out[1][self.k:]/self.Xfro_norm2 for out in outs]).sum() 

                    # undo the swap for the next iteration
                    indices_ij[j] = idx_j 
                    # unselected_inds_ij[i] = idx_i
            
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
                # chunk_inds = np.array_split(self.unselected_inds, self.unselected_inds.size//self.chunk_size)
                outs = Parallel(n_jobs=self.n_jobs)(delayed(nnls_OGM_gram)(self.G[np.ix_(self.indices, np.concatenate((self.indices, chunk)))], np.arange(self.k), G_diag=self.G_diag[np.concatenate((self.indices, chunk))], returnH=True) for chunk in self.chunk_inds)
                # self.H[self.indices, :] = np.eye(self.k)
                # self.H[self.unselected_inds, :self.k] = np.vstack([out[0].T for out in outs])
                self.H = np.vstack([out[0][:,self.k:].T for out in outs])

                # update the dists and energy values now that we've swapped
                self.dists = np.concatenate([out[1][self.k:]/self.Xfro_norm2 for out in outs])
                assert self.dists.size == self.n
                self.energy = self.dists.sum()
                self.energy_values.append(self.energy)
        
            else:
                continue_swap = False
                

            
        elif method[:2] == "p-" or method == "greedy":
            assert j_adap is not None 
            
            indices_wo_jadap = self.indices[:j_adap] + self.indices[j_adap+1:]
            # unselected_inds_wo_jadap = np.concatenate((self.unselected_inds, self.indices[j_adap]))
            outs = Parallel(n_jobs=self.n_jobs)(delayed(nnls_OGM_gram)(self.G[np.ix_(indices_wo_jadap, np.concatenate((indices_wo_jadap, chunk)))], np.arange(self.k-1), G_diag=self.G_diag[np.concatenate((indices_wo_jadap, chunk))], returnH=False) for chunk in self.chunk_inds)
            dists_wo_jadap = np.concatenate([out[1][self.k-1:]/self.Xfro_norm2 for out in outs])
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
                q_probs /= q_probs.sum()
                idx_j_new = np.random.choice(self.n, p=q_probs)
            
            # update object and corresponding energy
            self.indices[j_adap] = idx_j_new
            self.unselected_inds = np.delete(np.arange(self.n), self.indices)
            outs = Parallel(n_jobs=self.n_jobs)(delayed(nnls_OGM_gram)(self.G[np.ix_(self.indices, np.concatenate((self.indices, chunk)))], np.arange(self.k), G_diag=self.G_diag[np.concatenate((self.indices, chunk))], returnH=True) for chunk in self.chunk_inds)
            self.H = np.vstack([out[0][:,self.k:].T for out in outs])
            self.dists = np.concatenate([out[1][self.k:]/self.Xfro_norm2 for out in outs])
            self.energy = self.dists.sum()
            self.energy_values.append(self.energy)

        else:
            raise ValueError(f"swap_method = {method} not recognized for swap moves...")
        

        # return whether or not to continue swapping (based on if we've found a local minimizer in greedy setting)
        return continue_swap 