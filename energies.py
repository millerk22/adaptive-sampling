import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics.pairwise import rbf_kernel
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm 
from scipy.linalg import solve_triangular

IMPLEMENTED_ENERGIES = ['conic', 'cluster', 'cluster-dense']


class EnergyClass(object):
    def __init__(self, X, p=2):
        self.X = X
        self.k = None 
        self.p = p
        assert (self.p is None) or (self.p > 0)
        self.dim, self.n = X.shape
        self.energy = None
        self.indices = []
        self.dists = np.ones(self.n)
        self.energy_values = []
        self.type = None
        if sps.issparse(X):
            self.Xfro_norm2 = sps.linalg.norm(self.X, ord='fro')**2.
        else:
            self.Xfro_norm2 = np.linalg.norm(self.X, ord='fro')**2.

    def set_k(self, k):
        assert type(k) == int 
        assert k > 0
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
        self.compute_energy()
        return
    
    def swap(self, t, i):
        return

    def init_set(self, inds):
        assert self.k is None 
        self.set_k(len(inds))
        for i in inds:
            self.add(i)
        return

    def search_distances(self, candidates, idx_to_swap=None):
        return NotImplementedError()
    
    def compute_search_values(self, candidates=None, idx_to_swap=None):
        if candidates is None:
            candidates = np.delete(np.arange(self.n), self.indices)

        if idx_to_swap is not None:
            assert (idx_to_swap < len(self.indices))  and (idx_to_swap >= 0)
        
        search_dists = self.search_distances(candidates, idx_to_swap=idx_to_swap)
        
        # we make the assumption that we don't need to consider already selected points in our search values. We assume that their search 
        # value--even in the case of swaps--is simply the current energy.  
        all_search_vals = np.ones(self.n)*self.energy
        if self.p is not None:
            all_search_vals[candidates] = np.array([np.linalg.norm(s_dist, ord=self.p) for s_dist in search_dists])
        else:
            all_search_vals[candidates] = np.array([np.max(s_dist) for s_dist in search_dists])

        return all_search_vals
    
    def compute_swap_distances(self, idx_to_swap):
        return NotImplementedError()
    
    def compute_eager_swap_values(self, idx):
        return NotImplementedError()
    


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
        self.type = "conic"
        

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

    def compute_swap_distances(self, idx_to_swap):
        if len(self.indices) == 1:  # if we only have a single index in indices, we are going back to  sampling over the norms of each datapoint
            dists = np.linalg.norm(self.X, ord=2, axis=0).flatten()
        else:
            dists, _ = self.nnls_OGM_gram(idx_to_swap=idx_to_swap, returnH=False)
        if self.p is None:
            return 1.0*(dists == np.max(dists))   # mask where dists is largest
        return dists**(self.p)

    def swap(self, t, i):
        assert (t < len(self.indices))  and (t >= 0)
        self.indices[t] = i 
        self.G_S[t,:] = self.X.T @ self.X[:,i].flatten()
        dists, H = self.nnls_OGM_gram(returnH=True) 
        self.H = H   # assuming swap is only done with len(self.indices) = self.k
        self.dists = dists 
        self.compute_energy()
        self.energy_values.append(self.energy)
    
    def search_distances(self, candidates, idx_to_swap=None):
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
        
        
        
        return np.array(search_dists) 

    def compute_eager_swap_values(self, idx):
        if idx in self.indices:
            return np.ones(len(self.indices))*self.energy*(1.0000001)
        r = np.hstack([self.nnls_OGM_gram(search_ind=idx, idx_to_swap=j, returnH=False)[0].reshape(-1, 1) \
                      for j in range(len(self.indices))])
        if self.p is None:
            r = np.max(r, axis=0)
        else:
            r = np.linalg.norm(r, axis=0, ord=self.p)
        return r
        
    

    def nnls_OGM_gram(self, search_ind=None, idx_to_swap=None, delta=1e-6, maxiter=500, lam=1.0, returnH=True, verbose=False, term_cond=1):
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
            if search_ind is not None:
                S_ind_all[idx_to_swap] = search_ind 
            else:
                S_ind_all = S_ind_all[:idx_to_swap] + S_ind_all[idx_to_swap+1:]
        else:
            if search_ind is not None:
                S_ind_all = S_ind_all + [search_ind]
        
        if len(self.indices) == 0: # case of first iteration
            G_S = self.X[:, S_ind_all].T @ self.X
        else:
            if idx_to_swap is not None: 
                assert self.G_S.shape[0] == self.k  # swap move should only happen when self.G_S.shape[0] == self.k
                if search_ind is not None: 
                    G_S = np.array(self.G_S, copy=True) 
                    G_S[idx_to_swap, :] = self.X.T @ self.X[:,search_ind].flatten()
                else:
                    G_S = np.delete(self.G_S, idx_to_swap, axis=0)
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
        # if verbose:
        #     return {"dist_vals":dist_vals, "iters":i, "eps":eps, "eps0":eps0, "EPS":EPS }, H

        if verbose:
            with open(f"./results/conic_hull_energy_log_{self.n}.txt", "a") as f:
                f.write(f"{len(self.indices)},{self.k},{i}\n")
        if returnH:
            return dist_vals, H 
        
        return dist_vals, None




class LowRankEnergyDense(EnergyClass):
    """
    Currently writing the low-rank energy class... not considering complex valued matrices for now...
    """
    def __init__(self, X, p=2):
        super().__init__(X, p=p)
        self.dim, self.n = self.X.shape
        if self.n >= N_FLOAT_THRESHOLD:
            self.X = self.X.astype(np.float32)
        self.G = self.X.T @ self.X # precompute the Gram matrix  
        self.d = np.diagonal(self.G).copy()  # (self.dists**2.)
        self.dists = np.sqrt(self.d)
        self.compute_energy()
        self.W = None
        self.L = None
        self.f = None 
        self.type = "lowrank-dense"
        
    def set_k(self, k):
        super().set_k(k)
        self.W = np.zeros((self.k, self.n), dtype=self.X.dtype)
        self.L = np.zeros((self.k, self.k), dtype=self.X.dtype)
        self.f = np.zeros((self.k,), dtype=self.X.dtype)
    
    def add(self, i):   # update the interpolative decomposition (Algorithm 9.1)
        k_curr = len(self.indices)
        if k_curr == 0:
            # in the case of adding the first index, we just set these values directly
            self.L[0,0] = np.sqrt(self.G[i,i])
            self.f[0] = 1./self.G[i,i]
            self.W[0,:] = self.G[i,:] / self.G[i,i]
            self.d -= self.G[i,:]**2. / self.G[i,i]
        else:
            a = solve_triangular(self.L[:k_curr,:k_curr], self.G[self.indices,i], lower=True)
            v = self.G[i,i] - np.linalg.norm(a)**2.
            self.L[k_curr, :k_curr] = a 
            self.L[k_curr, k_curr] = np.sqrt(v)
            b = solve_triangular(self.L[:k_curr,:k_curr].T, a, lower=False)
            self.f[:k_curr] += b**2./v 
            self.f[k_curr] = 1./v
            r = self.G[i,:] - self.G[i,self.indices] @ self.W[:k_curr,:]
            self.W[k_curr,:] = r / v
            self.W[:k_curr,:] -= np.outer(b, r) / v
            self.d -= (r**2.) / v
        
        if hasattr(self, 'R'): # only if we're doing search builds
            self.R -= np.outer(self.W[k_curr,:], self.W[k_curr,:]) / self.f[k_curr]

        if np.absolute(self.d.min()) < -1e-9:
            print("something wrong, got a very negative value in d: ", self.d.min())
        
        self.d = np.clip(self.d, 0.0, None) # make sure we don't have negative values
        
        self.indices.append(i)
        self.dists = np.sqrt(self.d)   # get the non-squared Euclidean distances to the span
        self.compute_energy()
        self.energy_values.append(self.energy)
        return
    
    def downdate(self, t): # do in a bit, # t \in [0, len(self.indices)-1]
        
        k_curr = len(self.indices)
        assert t < k_curr and t >= 0
        if k_curr == 1:
            self.L = None
            self.W = None
            self.f = None
            self.d = np.diagonal(self.G)
        else:
            raise NotImplementedError("Downdate not implemented yet...")

        return 

    def search_distances(self, candidates, idx_to_swap=None):  # adaptive search build
        if idx_to_swap is None:
            if len(self.indices) == 0:
                self.R = self.G.copy()
            Q = np.outer(self.d, np.ones(len(candidates)))

            # don't want to consider those points who are already in the span for numerical stability reasons
            cand_outside_span = np.intersect1d(candidates, np.where(self.d > 1e-12)[0])
            cand_mask = np.isin(candidates, cand_outside_span)
            Q[:, cand_mask] -= (self.R[:, cand_outside_span] * self.R[:, cand_outside_span]) / self.d[np.newaxis, cand_outside_span]
            Q = np.clip(Q, 0.0, None)   
            Q = np.sqrt(Q)           # correct for fact that self.d has squared Euclidean distances
        else:
            raise ValueError("Shouldn't be using this function unless 'idx_to_swap=None'... something wrong")
        
        return Q.T    # return the transpose because of how search_distances is currently used in compute_search_values

    def swap(self, t, i):
        return 
    
    def compute_eager_swap_values(self, idx):  # adaptive search swaps
        return #r 

    def compute_swap_distances(self, idx_to_swap):  # adaptive sampling swap
        
        return #dists**(self.p)
    
    @staticmethod
    def cholesky_update(L, a): 
        """
        Perform a rank-1 Cholesky update of the L matrix with vector a.
        L is updated in place.
        """
        p = L.shape[0]
        for j in range(p):
            r = np.sqrt(L[j,j]**2. + a[j]**2.)
            c = L[j,j] / r
            s = a[j] / r
            L[j,j] = r
            L[j+1:,j] = c * L[j+1:,j] + s * a[j+1:]
            a[j+1:] = (a[j+1:] - s * L[j+1:,j]) / c

    @staticmethod
    def cholesky_downdate(L, a):
        """
        Perform a rank-1 Cholesky downdate of the L matrix with vector a.
        L is updated in place.
        """
        p = L.shape[0]
        for j in range(p):
            r = np.sqrt(L[j,j]**2. - a[j]**2.)
            c = r / L[j,j]
            s = a[j] / L[j,j]
            L[j,j] = r
            L[j+1:,j] = (L[j+1:,j] - s * a[j+1:]) / c
            a[j+1:] = c * a[j+1:] - s * L[j+1:,j]
    







######################################
############# NEED TO BE UPDATED #######
#####################################


N_FLOAT_THRESHOLD = 20000

class ClusteringEnergy(EnergyClass):
    """
    Need to debug and make sure it aligns with the fullD corresponding energy.
    """
    def __init__(self, X, p=2):
        super().__init__(X, p=p)
        self.dim, self.n = self.X.shape
        if self.n >= N_FLOAT_THRESHOLD:
            self.X = self.X.astype(np.float32)
        self.x_squared_norms = np.linalg.norm(self.X, axis=0)**2.
        self.dists = np.ones(self.n) # initial energy for adaptive sampling should give equal weight to every point
        self.compute_energy()
        self.type = "cluster"
    
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
            # dists = euclidean_distances(self.X[:,inds].reshape(1,-1), self.X.T, Y_norm_squared=self.x_squared_norms, \
            #                            squared=False).flatten()
            distances = cdist(self.X[:,inds].reshape(1,-1), self.X.T, metric='euclidean').flatten()
        elif type(inds) == list:
            #dists = euclidean_distances(self.X[:,inds].T, self.X.T, Y_norm_squared=self.x_squared_norms, \
                                        #  X_norm_squared=self.x_squared_norms[inds], squared=False)
            distances = cdist(self.X[:,inds].T, self.X.T, metric='euclidean')
        else:
            raise ValueError(f"inds must be of type `int` or `list`")
        
        return distances 
    
    def search_distances(self, candidates, idx_to_swap=None):
        if idx_to_swap is None:
            if len(self.indices) > 0:
                search_dists = [np.minimum(self.dists, self.compute_distances(c)) for c in candidates]
            else:
                search_dists = [self.compute_distances(c) for c in candidates]
        else:
            assert (0 <= idx_to_swap) and (idx_to_swap < len(self.indices)) 
            assert self.k == len(self.indices)
            if len(self.indices) == 1:
                search_dists = [self.compute_distances(c) for c in candidates]
            else:
                dists_wo_st = np.min(np.vstack((self.D[:idx_to_swap,:], self.D[idx_to_swap+1:,:])), axis=0) 
                search_dists = [np.minimum(dists_wo_st, self.compute_distances(c)) for c in candidates]
        return np.array(search_dists)

    def compute_swap_distances(self, idx_to_swap):
        if len(self.indices) == 1:  # if we only have a single index in indices, we are going back to uniform sampling over the dataset
            return np.ones(self.n)
        inds_wo_t = self.indices[:]
        inds_wo_t.pop(idx_to_swap)
        dists = self.compute_distances(inds_wo_t).min(axis=0).flatten()
        if self.p is None:
            return 1.*(dists == np.max(dists))
        return dists**(self.p)
    
    def compute_eager_swap_values(self, idx):
        raise NotImplementedError("Not implemented yet...")


'''
self.dists = q_S vector from the paper
'''

class ClusteringEnergyDense(EnergyClass):
    def __init__(self, X, p=2):
        super().__init__(X, p=p)
        self.dim, self.n = self.X.shape
        if self.n >= N_FLOAT_THRESHOLD:
            self.X = self.X.astype(np.float32)
        self.x_squared_norms = np.linalg.norm(self.X, axis=0)**2.
        self.dists = np.ones(self.n) # initial energy for adaptive sampling should give equal weight to every point
        self.compute_energy()
        self.D = self.compute_distances()  
        self.q2 = None 
        self.h = None
        self.type = "cluster-dense"
    
    def set_k(self, k):
        super().set_k(k)
        
    def add(self, i):
        self.indices.append(i)
        if len(self.indices) == 1:
            # when adding the first index, we just set dists to be the distances to that point
            self.dists = self.D[:,i].copy()
            self.h = np.zeros((self.n,))
        else:
            # when adding subsequent indices, we need to update dists and q2
            mask = self.D[:,i] < self.dists 
            if self.q2 is None:
                # when have only 2 indices, we can simply calculate q2 in terms of mask and dists
                self.q2 = self.dists.copy()  
                self.q2[~mask] = self.D[~mask,i].copy()
            else:
                # when have more than 2 indices, we use another mask to consider the case when d_i is less than q2,
                # necessitating an update of q2 on these indices
                mask2 = ~mask & (self.D[:,i] < self.q2)
                self.q2[mask2] = self.D[mask2, i].copy()
            
            # update of h and dists only happens where mask is true. Do this after the update to q2 since we might need to use 
            # old value of dists to update q2
            self.h[mask] = len(self.indices) - 1
            self.dists[mask] = self.D[mask,i].copy()

        # compute the energy            
        self.compute_energy()
        self.energy_values.append(self.energy)
        return 
    

    def swap(self, t, i):
        # case of self.q2 is None is currently handled
        assert (t < len(self.indices))  and (t >= 0)
        Vor_mask = self.h == t # in the Voronoi cell of the point being swapped out
        mask1 = self.D[:,i] < self.dists 
        mask2 = self.D[:,i] < self.q2 
        notVor1 = ~Vor_mask & mask1 # outside t's Voronoi cell and d_i(j) < q_S(j)
        notVor2 = ~Vor_mask & ~mask1 & mask2 # outside t's Voronoi cell and q_S(j) <= d_i(j) < q2(j)
        Vor1 = Vor_mask & mask2 # inside t's Voronoi cell and d_i(j) < q2(j)
        Vor2 = Vor_mask & ~mask2 # inside t's Voronoi cell and  d_i(j) >= q2(j)

        
        # Something wrong with Vor2 computations...
        self.Vor1 = Vor1.copy()
        self.Vor2 = Vor2.copy()
        self.notVor1 = notVor1.copy()
        self.notVor2 = notVor2.copy()

        #### outside t's Voronoi cell updates
        self.q2[notVor2] = self.D[notVor2, i]  # Case II: only update q2 where q_S(j) <= d_i(j) < q2(j)
        self.q2[notVor1] = self.dists[notVor1] # Case I: update dists, q2, h where d_i(j) < q_S(j)
        self.dists[notVor1] = self.D[notVor1, i]
        self.h[notVor1] = t 

        
        self.indices[t] = i
        
        #### inside t's Voronoi cell updates
        self.dists[Vor1] = self.D[Vor1,i] # Case I: update only dists where d_i(j) < q_S(j). h and q2 stay unchanged
        # Case II: need to "fully" update dists, q2, and h
        numV2 = Vor2.sum() 
        Dv2 = self.D[np.ix_(Vor2,self.indices)]  # self.indices now contains the inserted point, x_i
        h1h2 = np.argsort(Dv2,axis=1)[:,:2]  # indices of the two smallest distances in each row of Dv2
        self.h[Vor2] = h1h2[:,0]
        self.dists[Vor2] = Dv2[np.arange(numV2),h1h2[:,0]].flatten()
        self.q2[Vor2] = Dv2[np.arange(numV2),h1h2[:,1]].flatten()
        
        self.compute_energy()
        self.energy_values.append(self.energy)
    
    def compute_distances(self, inds=None):
        if type(inds) in [int, np.int8, np.int16, np.int32, np.int64]:
            distances = cdist(self.X[:,inds].reshape(1,-1), self.X.T, metric='euclidean').flatten()
        elif type(inds) == list:
            distances = cdist(self.X[:,inds].T, self.X.T, metric='euclidean')
        elif inds is None:
            distances = squareform(pdist(self.X.T, metric='euclidean'))
        else:
            raise ValueError(f"inds must be of type `int` or `list`")
        
        return distances 
    
    def search_distances(self, candidates, idx_to_swap=None):
        if idx_to_swap is None:
            if len(self.indices) == 0:
                search_dists = self.D[:,candidates].T
            else:
                search_dists = np.array([np.minimum(self.dists, self.D[:,c]) for c in candidates])
        else:
            raise ValueError("Shouldn't be using this function unless 'idx_to_swap=None'... something wrong")
        
        return search_dists

    def compute_swap_distances(self, idx_to_swap):
        if len(self.indices) == 1:  # if we only have a single index in indices, we are going back to uniform sampling over the dataset
            return np.ones(self.n)
        inds_wo_t = self.indices[:]
        inds_wo_t.pop(idx_to_swap)
        dists = self.D[:,inds_wo_t].min(axis=1).flatten()
        if self.p is None:
            return 1.*(dists == np.max(dists))
        return dists**(self.p)
    
    def compute_eager_swap_values(self, idx):
        if idx in self.indices:
            return np.ones(len(self.indices))*self.energy*(1.0000001)
        Dtilde = self.D[:,self.indices+[idx]]
        r = np.vstack([np.min(np.hstack((Dtilde[:,:j], Dtilde[:,j+1:])), axis=1).reshape(1,-1) for j in range(len(self.indices))])
        if len(r.shape) == 1:
            r = r.reshape(1,-1)
        if self.p is None:
            r = np.max(r, axis=1)
        else:
            r = np.linalg.norm(r, axis=1, ord=self.p)
        
        return r 

    def compute_C_matrix(self):
        '''
        obsolete with how we're doing search now...
        '''
        if self.p is None:
            p_  = np.inf
        else:
            p_ = self.p
        if len(self.indices) == 1: # in this case, we should just be returning C \in R^{n x 1}, C(i,1) = f({x_i}), which can be computed easily from self.D
            return  np.linalg.norm(self.D, axis=1, ord=p_).reshape(-1,1)
        ns = np.argsort(self.D[:, self.indices], axis=1)
        n = ns[:,0]
        nr = ns[:,1]
        q = np.take_along_axis(self.D[:,self.indices], np.expand_dims(n, axis=1), axis=1).squeeze(axis=1)
        r = np.take_along_axis(self.D[:,self.indices], np.expand_dims(nr, axis=1), axis=1).squeeze(axis=1)
        Q = np.minimum(self.D, q.reshape(-1,1))
        C = np.linalg.norm(Q, axis=0, ord=p_).reshape(-1,1) * np.ones(len(self.indices)).reshape(1,-1)
        for l in range(self.n):
            if self.p is not None:
                C[:,n[l]] = (C[:,n[l]]**self.p  + np.minimum(self.D[:,l], r[l])**self.p - Q[l,:]**self.p)**(1./self.p)  # update all the l^th terms in the sums corresponding to ejecting n[l]th current prototype
            else:
                C[:,n[l]] = np.maximum(C[:,n[l]], np.minimum(self.D[:,l], r[l]))
        
        return C
        





class ClusteringEnergyDenseOld(EnergyClass):
    """
    Need to finish debugging this...
    """
    def __init__(self, X, p=2):
        super().__init__(X, p=p)
        self.dim, self.n = self.X.shape
        if self.n >= N_FLOAT_THRESHOLD:
            self.X = self.X.astype(np.float32)
        self.x_squared_norms = np.linalg.norm(self.X, axis=0)**2.
        self.dists = np.ones(self.n) # initial energy for adaptive sampling should give equal weight to every point
        self.compute_energy()
        #print("Computing full Distance and Quality matrices up front...\n\tRecommended to only use this for search build and search/sampling swap phases...")
        self.D = self.compute_distances()  
        self.Q = self.D.copy()
        self.type = "cluster-dense"
    
    def set_k(self, k):
        super().set_k(k)
        
    def add(self, i):
        self.indices.append(i)
        self.dists = self.Q[:,self.indices[-1]]                    # updated dists are just the most recent column of Q
        np.minimum(self.dists.reshape(-1, 1), self.D, out=self.Q)  # update the quality matrix
        self.compute_energy()
        self.energy_values.append(self.energy)
        return 
    

    def swap(self, t, i):
        assert (t < len(self.indices))  and (t >= 0)
        self.indices[t] = i 
        self.dists = self.D[:,self.indices].min(axis=1).flatten()
        self.compute_energy()
        self.energy_values.append(self.energy)
    
    def compute_distances(self, inds=None):
        if type(inds) in [int, np.int8, np.int16, np.int32, np.int64]:
            # dists = euclidean_distances(self.X[:,inds].reshape(1,-1), self.X.T, Y_norm_squared=self.x_squared_norms, \
            #                            squared=False).flatten()
            dists = cdist(self.X[:,inds].reshape(1,-1), self.X.T, metric='euclidean').flatten()
        elif type(inds) == list:
            # dists = euclidean_distances(self.X[:,inds].T, self.X.T, Y_norm_squared=self.x_squared_norms, \
            #                              X_norm_squared=self.x_squared_norms[inds], squared=False)
            dists = cdist(self.X[:,inds].T, self.X.T, metric='euclidean')
        elif inds is None:
            # dists = euclidean_distances(self.X.T, self.X.T, Y_norm_squared=self.x_squared_norms, \
            #                              X_norm_squared=self.x_squared_norms, squared=False)

            dists = squareform(pdist(self.X.T, metric='euclidean'))
        else:
            raise ValueError(f"inds must be of type `int` or `list`")
        
        return dists 
    
    def search_distances(self, candidates, idx_to_swap=None):
        if idx_to_swap is None:
            search_dists = self.Q[:,candidates].T
        else:
            raise ValueError("Shouldn't be using this function unless 'idx_to_swap=None'... something wrong")
        return search_dists

    def compute_swap_distances(self, idx_to_swap):
        if len(self.indices) == 1:  # if we only have a single index in indices, we are going back to uniform sampling over the dataset
            return np.ones(self.n)
        inds_wo_t = self.indices[:]
        inds_wo_t.pop(idx_to_swap)
        dists = self.D[:,inds_wo_t].min(axis=1).flatten()
        if self.p is None:
            return 1.*(dists == np.max(dists))
        return dists**(self.p)
    
    def compute_eager_swap_values(self, idx):
        Dtilde = self.D[:,self.indices+[idx]]
        r = np.hstack([np.min(np.hstack((Dtilde[:,:j], Dtilde[:,j+1:])), axis=1) for j in range(len(self.indices))])
        
        if len(r.shape) == 1:
            r = r.reshape(1,-1)

        if self.p is None:
            r = np.max(r, axis=1)
        else:
            r = np.linalg.norm(r, axis=1, ord=self.p)
        
        return r 

    def compute_C_matrix(self):
        '''
        obsolete with how we're doing search now...
        '''
        if self.p is None:
            p_  = np.inf
        else:
            p_ = self.p
        if len(self.indices) == 1: # in this case, we should just be returning C \in R^{n x 1}, C(i,1) = f({x_i}), which can be computed easily from self.D
            return  np.linalg.norm(self.D, axis=1, ord=p_).reshape(-1,1)
        ns = np.argsort(self.D[:, self.indices], axis=1)
        n = ns[:,0]
        nr = ns[:,1]
        q = np.take_along_axis(self.D[:,self.indices], np.expand_dims(n, axis=1), axis=1).squeeze(axis=1)
        r = np.take_along_axis(self.D[:,self.indices], np.expand_dims(nr, axis=1), axis=1).squeeze(axis=1)
        Q = np.minimum(self.D, q.reshape(-1,1))
        C = np.linalg.norm(Q, axis=0, ord=p_).reshape(-1,1) * np.ones(len(self.indices)).reshape(1,-1)
        for l in range(self.n):
            if self.p is not None:
                C[:,n[l]] = (C[:,n[l]]**self.p  + np.minimum(self.D[:,l], r[l])**self.p - Q[l,:]**self.p)**(1./self.p)  # update all the l^th terms in the sums corresponding to ejecting n[l]th current prototype
            else:
                C[:,n[l]] = np.maximum(C[:,n[l]], np.minimum(self.D[:,l], r[l]))
        
        return C
        



