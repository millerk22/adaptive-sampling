import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics.pairwise import rbf_kernel
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm 
from scipy.linalg import solve_triangular


IMPLEMENTED_ENERGIES = ['cluster', 'lowrank', 'conic', 'convex']
N_FLOAT_THRESHOLD = 20000

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
        self.type = None

    def set_k(self, k):
        assert type(k) == int 
        assert k > 0
        assert len(self.indices) == 0
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
    
    def swap(self, t, i, debug=False):
        return

    def init_set(self, inds, return_values=False):
        assert self.k is None 
        self.set_k(len(inds))
        vals = [self.energy]
        for i in inds:
            self.add(i)
            vals.append(self.energy)
        if return_values:
            return vals
        return

    def search_distances(self, candidates, idx_to_swap=None):
        return NotImplementedError()
    
    def compute_search_values(self, candidates=None):
        if candidates is None:
            candidates = np.delete(np.arange(self.n), self.indices)
        
        search_dists = self.search_distances(candidates)
        
        # weassume that we don't need to consider already selected points in our search values. We assume that their search 
        # value--even in the case of swaps--is simply the current energy.  
        all_search_vals = np.ones(self.n)*self.energy
        if self.p is not None:
            all_search_vals[candidates] = np.linalg.norm(search_dists, ord=self.p, axis=1)
        else:
            all_search_vals[candidates] = np.max(search_dists, axis=1)

        return all_search_vals
    
    def compute_swap_distances(self, idx_to_swap):
        return NotImplementedError()
    
    def compute_eager_swap_values(self, idx):
        return NotImplementedError()

    def prep_for_swaps(self, method="sampling"):
        return 
    
    def end_swaps(self):
        return 
    




class ClusteringEnergy(EnergyClass):
    def __init__(self, X, p=2):
        super().__init__(X, p=p)
        self.dim, self.n = self.X.shape
        if self.n >= N_FLOAT_THRESHOLD:
            self.X = self.X.astype(np.float32)
        self.x_squared_norms = np.linalg.norm(self.X, axis=0)**2.
        self.dists = np.ones(self.n) # initial energy for adaptive sampling should give equal weight to every point
        self.compute_energy()
        self.D = self.compute_distances()   
        self.near = None 
        self.next_near = None 
        self.next_dists = None
        self.type = "cluster"
    
    def set_k(self, k):
        super().set_k(k)
        
    def add(self, i):
        if len(self.indices) == 0:
            # when adding the first index, we just set dists to be the distances to that point
            self.dists = self.D[:,i].copy()
            self.near = np.zeros((self.n,), dtype=int)
        else:
            closest_to_i = self.D[:,i] < self.dists
            self.dists[closest_to_i] = self.D[closest_to_i,i]
            self.near[closest_to_i] = len(self.indices)

        self.indices.append(i)
        # compute the energy            
        self.compute_energy()
        return 
    

    def swap(self, t, i, debug=False):
        '''
        For each x_j in dataset, let: d_i = d(x_j, x_i), d_near = d(x_j, Y), d_next_near = d(x_j, Y \ {nearest proto to x_j})
        
        Regions:  d_i \in I, II, or III 
        |     I         |       II          |          III
        |------------d_near-------------d_next_near--------------
        
        Cases:[A] x_j in V_t  --> d_near is removed
              [B] x_j in V_t2 --> d_next_near is removed 
              [C] o.w.        --> neither is removed

        Results:  
            |             I              |            II                |              III               |
            ---------------------------------------------------------------------------------------------
         A  |                   replace d_near = d_i                    |  recompute near,next_near
            |---------------------------------------------------------------------------------------------
         B  | d_next_near = d_near,      |    d_next_near = d_i         |    recompute next_near         |
            |    d_near = d_i            |                              |                                |
            |---------------------------------------------------------------------------------------------
         C  | d_next_near = d_near,      |     d_next_near = d_i        |            NOTHING             |
            |   d_near = d_i             |                              |                                |
            ----------------------------------------------------------------------------------------------
        '''

        if len(self.indices) == 0:
            raise ValueError("Cannot swap with no prototypes chosen.")
        assert (t < len(self.indices))  and (t >= 0)  

        if i in self.indices:
            print(f"Warning: {i} already in self.indices -- not a valid swap. Skipping...")
            return 

        if len(self.indices) == 1:
            # simple case, just switch dists with new i dists (self.near is already all zeros, nothing to change)
            self.dists = self.D[:,i].copy()
            self.indices[t] = i
            self.compute_energy()
            return 
        
        if self.next_near is None: # ensure we are prepped to swap
            self.prep_for_swaps()

        d_i = self.D[:,i].copy()  

        # Iteratively refine the partition of indices based on the cases/regions above
        all_idxs = np.arange(self.n)

        regIII = np.where(self.next_dists < d_i)[0]
        regI_II = np.setdiff1d(all_idxs, regIII)
        regI_II_caseA = regI_II[self.near[regI_II] == t]
        regI_II_caseB_C = np.setdiff1d(regI_II, regI_II_caseA) 
        regI_caseB_C = regI_II_caseB_C[d_i[regI_II_caseB_C] < self.dists[regI_II_caseB_C]]
        regII_caseB_C = np.setdiff1d(regI_II_caseB_C, regI_caseB_C)
        regIII_caseA_B = regIII[(self.near[regIII] == t) | (self.next_near[regIII] == t)]
        
        # Region I and II, Case A: just update nearest dists with d_i. (near remains t)
        self.dists[regI_II_caseA] = d_i[regI_II_caseA]

        # Region I, Cases B and C: update dists, next_dists, near, and next_near
        self.next_dists[regI_caseB_C] = self.dists[regI_caseB_C]
        self.next_near[regI_caseB_C] = self.near[regI_caseB_C]
        self.dists[regI_caseB_C] = d_i[regI_caseB_C]
        self.near[regI_caseB_C] = t

        # Region II, Cases B and C: update next_dists, but don't need to update next_near (remains t)
        self.next_dists[regII_caseB_C] = d_i[regII_caseB_C]
        
        # Region III, Cases A and B: update everything from scratch
        self.indices[t] = i   # need to replace tth prototype with x_i
        if regIII_caseA_B.size > 0:
            self.near[regIII_caseA_B], self.dists[regIII_caseA_B], self.next_near[regIII_caseA_B], self.next_dists[regIII_caseA_B] = \
                        ClusteringEnergy.first_second_smallest_per_row(self.D[np.ix_(regIII_caseA_B, self.indices)])
        
        self.compute_energy()
        return 
        
    
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
    
    def search_distances(self, candidates):
        # Q[i,:] = q_{+i} vector in Alg 8.2
        if len(self.indices) == 0:
            Q = self.D[:,candidates].T
        else:
            Q = np.minimum(self.dists, self.D[candidates,:])
        return Q

    def compute_swap_distances(self, idx_to_swap):
        if len(self.indices) == 1:  # if we only have a single index in indices, we are going back to uniform sampling over the dataset
            return np.ones(self.n)
        inds_wo_t = self.indices[:]
        inds_wo_t.pop(idx_to_swap)
        dists = self.D[:,inds_wo_t].min(axis=1).flatten()
        if self.p is None:
            return 1.*(dists == np.max(dists))
        return (dists/dists.max())**(self.p)
    
    def compute_eager_swap_values(self, idx): # Alg. 8.4
        if len(self.indices) == 1:
            d_i = self.D[:,idx].copy()
            if self.p is not None:
                return np.array([np.linalg.norm(d_i, ord=self.p)])
            else:
                return np.array([np.max(d_i)])
            
        if idx in self.indices: # should be ignored
            return np.ones(len(self.indices))*self.energy*(1.1)
        
        Delta = np.zeros(len(self.indices), dtype=float)
        cost_change = 0 
        d_i = self.D[:,idx].copy()
        
        if self.p is not None:
            regI = d_i < self.dists
            regII = (~regI) & (d_i < self.next_dists)
            regIII = ~(regI | regII)
            
            cost_change = np.sum(d_i[regI]**self.p - self.dists[regI]**self.p)

            # precompute the per-point changes from regions II and III
            changeII = d_i[regII]**self.p - self.dists[regII]**self.p
            changeIII = self.next_dists[regIII]**self.p - self.dists[regIII]**self.p

            # accumulate changes into Delta for each prototype
            np.add.at(Delta, self.near[regII], changeII)
            np.add.at(Delta, self.near[regIII], changeIII)

            Delta = Delta + cost_change

            #### Need to compare against current energy in the sampler given current implementation
            Delta += self.energy**self.p 
            Delta = Delta**(1./self.p)

        else: # p = \infty case isn't as straightforward, we'll just do normal way here
            Dtilde = self.D[:,self.indices+[idx]]
            Delta = np.vstack([np.min(np.hstack((Dtilde[:,:j], Dtilde[:,j+1:])), axis=1).reshape(1,-1) for j in range(len(self.indices))])
            if len(Delta.shape) == 1:
                Delta = Delta.reshape(1,-1)
            Delta = Delta.max(axis=1) 
        
        return Delta

    def prep_for_swaps(self, method="sampling", debug=False):
        if self.k > 2:
            near1, d1, near2, d2 = ClusteringEnergy.first_second_smallest_per_row(self.D[:,self.indices])
            if debug:
                assert np.allclose(d1, self.dists)
                assert np.allclose(near1, self.near)
            self.next_near = near2                               # next-nearest prototype indices
            self.next_dists = d2                                 # next-nearest distances 
        elif self.k == 2:
            self.next_near = 1 - self.near 
            self.next_dists = np.take_along_axis(self.D, np.array(self.indices)[self.next_near, None], axis=1).flatten()

        return 
    
    @staticmethod
    def first_second_smallest_per_row(data):
        if data.ndim != 2:
            raise ValueError("data must be 2D (n, k).")
        n, k = data.shape
        if k < 2:
            raise ValueError("Need at least 2 columns to get first and second smallest.")

        # Get indices of the two smallest (unordered) per row
        idx2cand = np.argpartition(data, kth=1, axis=1)[:, :2]   # (n, 2)

        rows = np.arange(n)[:, None]
        vals2cand = data[rows, idx2cand]                         # (n, 2)

        # Order the two candidates so column 0 is smallest, column 1 is second smallest
        swap = vals2cand[:, 0] > vals2cand[:, 1]
        idx_sorted = idx2cand.copy()
        val_sorted = vals2cand.copy()
        idx_sorted[swap] = idx_sorted[swap][:, ::-1]
        val_sorted[swap] = val_sorted[swap][:, ::-1]

        idx1, idx2 = idx_sorted[:, 0], idx_sorted[:, 1]
        val1, val2 = val_sorted[:, 0], val_sorted[:, 1]

        return idx1, val1, idx2, val2
    



class LowRankEnergy(EnergyClass):
    """
    Low-rank energy class that supports both real and complex valued matrices.
    For complex X, we work with the Hermitian Gram matrix G = X^* X.
    """
    def __init__(self, X, p=2):
        super().__init__(X, p=p)
        self.dim, self.n = self.X.shape

        # Detect whether X is complex
        self.is_complex = np.iscomplexobj(self.X)

        # Downcast only in the purely real case (preserve complex dtype otherwise)
        if (not self.is_complex) and (self.n >= N_FLOAT_THRESHOLD):
            self.X = self.X.astype(np.float32)

        # Compute the Gram matrix G = X^* X
        if self.is_complex:
            self.G = self.X.conj().T @ self.X
        else:
            self.G = self.X.T @ self.X

        self.d = np.real(np.diagonal(self.G).copy())
        self.dists = np.sqrt(self.d)
        self.compute_energy()
        self.W = None   # interpolation weights
        self.L = None   # Cholesky factor of G[indices, indices]
        self.f = None   # diagonal of G[indices, indices]^{-1}
        self.type = "lowrank"
        self.U = None  # for adaptive sampling swap
        self.verbose = False
        self.test = False
        
    def set_k(self, k):
        super().set_k(k)
        assert(self.W is None and self.L is None and self.f is None), "Cannot call set_k after objects have been allocated."
        # W and L follow X's dtype; f is always real
        self.W = np.zeros((self.k, self.n), dtype=self.X.dtype)
        self.L = np.zeros((self.k, self.k), dtype=self.X.dtype)
        self.f = np.zeros((self.k,))
        return 
    
    def prep_for_swaps(self, method="sampling"):
        if method == "search":
            p2case = False 
            if self.p is not None:
                if np.isclose(self.p, 2) and not self.test:
                    p2case = True
            if p2case:
                if not hasattr(self, 'R'): # if did not have a search build previously, compute R here 
                    self.R = self.G - self.W.conj().T @ (self.G[np.ix_(self.indices,self.indices)] @ self.W)
                self.Y = self.W @ self.R # compute Y = W R upfront, updates will be easy thereafter 
            else:
                if self.indices[-1] == -1:
                    print("WARNING.... Looks like you're calling prep_for_swaps a second time, which can cause problems..")
                # pad W, L, f with zeros to allow for temporary (k+1) prototypes during search swap computations
                self.W = np.vstack((self.W, np.zeros((1, self.n), dtype=self.X.dtype)))
                self.L = np.vstack((self.L, np.zeros((1, self.L.shape[1]), dtype=self.X.dtype)))
                self.L = np.hstack((self.L, np.zeros((self.L.shape[0], 1), dtype=self.X.dtype)))
                self.L[-1,-1] = 1.0
                self.f = np.concatenate((self.f, np.array([0.0])))
                self.indices = self.indices + [-1]  # placeholder for temporary (k+1) prototype
            
        return 
    
    def end_swaps(self):
        if self.L.shape[0] == self.k+1:
            # remove the temporary (k+1) prototype data at the end of search swap computations
            self.W = self.W[:-1,:]
            self.L = self.L[:-1,:-1]
            self.f = self.f[:-1]
            self.indices = self.indices[:-1]
        return

    def add(self, i):   # update the interpolative decomposition (Algorithm 9.1)
        k_curr = len(self.indices)
        G_ii = np.real(self.G[i, i])
        if k_curr == 0:
            self.L[0, 0] = np.sqrt(G_ii)
            self.f[0] = 1.0 / G_ii
            self.W[0, :] = self.G[i, :] / G_ii
            self.d -= np.abs(self.G[i, :])**2 / G_ii

        else:
            # Solve L a = G[indices, i]
            a = solve_triangular(self.L[:k_curr, :k_curr], self.G[self.indices, i], lower=True)
            v = G_ii - np.vdot(a, a).real
            if np.isclose(v, 0.0) or v < 0.0:
                print(f"WARNING: Adding {i} to the decomposition resulted in non-positive v = {v} (i.e., singular G[S, S])...")
                print("\tNOT ADDING...")
                return 

            # update L
            self.L[k_curr, :k_curr] = a.conj()
            self.L[k_curr, k_curr] = np.sqrt(v)

            # Solve L^* b = a  (conjugate transpose)
            b = solve_triangular(self.L[:k_curr, :k_curr], a, lower=True, trans='C')

            # update f
            self.f[:k_curr] += (np.abs(b)**2) / v
            self.f[k_curr] = 1.0 / v

            # update W
            r = self.G[i, :] - self.G[i, self.indices] @ self.W[:k_curr, :]
            r[i] = v
            r_v = r / v
            self.W[k_curr, :] = r_v
            self.W[:k_curr, :] -= np.outer(b, r_v) 
            
            # update d
            dold = np.copy(self.d)
            self.d -= np.abs(r)**2 / v
            min_idx = np.argmin(self.d)
        
        # Update R if we're in a search build
        if hasattr(self, 'R'):
            w_k = self.W[k_curr, :]
            self.R -= np.outer(w_k, w_k.conj()) / self.f[k_curr]

        # Sanity check: d should not be significantly negative
        if self.d.min() < -1e-9:
            print("Something wrong, got a very negative value in d: ", self.d.min())
            print("\tPerhaps increase precision to float64...")
        # Clip small negatives from numerical errors
        self.d = np.clip(self.d, 0.0, None)
        
        self.indices.append(i)
        self.dists = np.sqrt(self.d)   # Euclidean distances to the span
        self.compute_energy()
        
        return

    def search_distances(self, candidates):  # adaptive search build
        if len(self.indices) == 0:
            self.R = self.G.copy()
        elif not hasattr(self, 'R'):
            submatrix = self.G[np.ix_(self.indices,self.indices)]
            self.R = self.G - self.W[:len(self.indices),:].conj().T @ submatrix @ self.W[:len(self.indices),:]
            
        Q = np.outer(np.ones(len(candidates)), self.d)

        # don't want to consider those points already in the span for numerical stability reasons
        cand_outside_span = np.intersect1d(candidates, np.where(self.d > 1e-12)[0])
        cand_mask = np.isin(candidates, cand_outside_span)

        # compute update of distances for all candidates outside the span
        if cand_outside_span.size > 0:
            Q[cand_mask, :] -= (np.abs(self.R[cand_outside_span,:])**2) / self.d[cand_outside_span, np.newaxis]

        Q = np.clip(Q, 0.0, None)
        Q = np.sqrt(Q)  # because self.d stores squared distances, and we want Euclidean distances

        # return transpose because of how search_distances is currently used in compute_search_values
        return Q
    
    def downdate(self, t): # (Algorithm 9.4)
        k_max = self.f.size
        assert t < k_max and t >= 0
        wt = np.copy(self.W[t,:])
        ft = np.copy(self.f[t])

        # downdate d first
        self.d += np.abs(wt)**2 / ft
        
        # perform Cholesky Delete operation
        self.cholesky_delete(self.L, t)

        # downdate W and f
        a = solve_triangular(self.L, self.G[self.indices, self.indices[t]], lower=True) 
        b = solve_triangular(self.L, a, lower=True, trans='C')
        b[t] = -1.0
        self.W  += np.outer(b, wt)
        a_t = np.concatenate((a[:t], a[t+1:]))
        self.f -= np.abs(b)**2 / (self.G[self.indices[t], self.indices[t]].real - np.vdot(a_t,a_t).real)
        
        # is this going to be bad to do? Seems like we have a problem with numerical instability if we don't do this...
        self.f[t] = 0.0  
        self.W[t, :] = 0.0
        
        # update these values
        self.dists = np.sqrt(self.d)  
        self.compute_energy()


        # if have attribute Y and R (p=2 search swap), downdate accordingly 
        if hasattr(self, 'Y'):
            self.R += np.outer(wt.conj(), wt)/ft 
            self.Y += np.outer(b, self.Y[t,:]) + np.outer(self.W @ wt.conj(), wt) / ft

        return 
    
    def update(self, t, i): # (Algorithm 9.5)
        '''
        __Note:__ This update function requires the decomposition to be prepared for an update:
            * L has a 1.0 at (t,t) and zeros below it in column t and zeros in row t
            * W has zeros in row t
            * f has a 0.0 at index t
        This is because we assume that either (1) a downdate has already been performed at index t 
        or (2) we are performing an update after augmenting the decomposition for search swap moves.

        Furthermore, the t^th value in self.indices will be __ignored__ and replaced with i during the update. But 
        this function is written so that it does not actually remove the t^th index from self.indices; it simply overwrites it.
        '''
        # t is the index in self.indices to be replaced with i (t = i, i = s_i' in the notation of the paper)
        k_max = self.f.size
        assert t < k_max and t >= 0

        # check the the decomposition is ready for the update (i.e., has does not have an active prototype at index t)
        assert self.L[t,t] == 1.0
        assert np.isclose(self.L[t+1:,t], 0.0).all()
        assert np.isclose(self.W[t, :], 0.0).all()
        assert np.isclose(self.f[t], 0.0)
        
        # solve for some auxiliary variables
        a = solve_triangular(self.L, self.G[self.indices, i], lower=True)
        a_t = np.concatenate((a[:t], a[t+1:]))
        v = self.G[i,i].real - np.vdot(a_t, a_t).real
        if np.isclose(v, 0.0) or v < 0.0:
            if self.verbose:
                print(f"WARNING: Updating to add {i} to the decomposition resulted in non-positive v = {v} (i.e., singular G[S, S])...")
                print("\tNOT UPDATING, reverting to previous prototypes...")
            
            # revert back and don't perform the swap
            return self.update(t, self.indices[t]) 
            
        b = solve_triangular(self.L, a, lower=True, trans='C')
        b[t] = -1.0

        # update f, W, and d
        self.f += np.abs(b)**2. / v
        rstar = self.G[i,:] -  self.G[i, self.indices] @ self.W

        if hasattr(self, 'Y'):
            # compute this prior to update self.W
            W_Wprimestar = self.W @ (self.W[t,:].conj() - b[t].conj() * rstar.conj()/v) 

        self.W -= np.outer(b, rstar)/ v
        self.d -= np.abs(rstar)**2. / v

        np.clip(self.d, 0.0, None, out=self.d)  # clip small negatives from numerical errors

        # update indices and change the cholesky factor via add
        self.indices[t] = i
        self.cholesky_add(self.L, self.G[self.indices, i] , t)

        # update energy's values 
        self.dists = np.sqrt(self.d)   
        self.compute_energy()
        
        # if have attribute Y and R (p=2 search swap), downdate accordingly 
        if hasattr(self, 'Y'):
            self.R -= np.outer(self.W[t,:].conj(), self.W[t,:]) / self.f[t]
            self.Y -= np.outer(W_Wprimestar, self.W[t,:])/self.f[t] + np.outer(b, (rstar[np.newaxis, :] @ self.R).flatten()) / v

        return 

    def prep_all_downdates(self, returnU=False):
        U = np.tile(self.d, (self.f.size, 1)) + np.abs(self.W)**2 / self.f[:, np.newaxis] # do f.size in case of search swap prep, where we have not overwritten self.k...
        U = np.clip(U, 0.0, None)
        if self.p is None:
            self.U = np.sqrt(U)
        elif np.isclose(self.p, 2):
            self.U = U
        else:
            self.U = U**(0.5*self.p)

        if returnU:
            return self.U
        return 
    
    def swap(self, t, i, debug=False):
        if i in self.indices[:self.k]:
            print(f"Warning: {i} already in self.indices -- not a valid swap. Skipping...")
            return 
        
        if debug and self.verbose:
            print(f"before downdate to swap at {t}, {self.indices[t]} and {i} :")
            print("L: ", np.round(self.L, 1))
            print("W: ", np.round(self.W, 1))
            print("f: ", np.round(self.f, 1))
            print("d: ", np.round(self.d, 1))
            print("indices: ", self.indices)

        self.downdate(t)

        if debug and self.verbose:
            print("after downdate, before interchange/update:")
            print("L: ", np.round(self.L, 1))
            print("W: ", np.round(self.W, 1))
            print("f: ", np.round(self.f, 1))
            print("d: ", np.round(self.d, 1))
            print("indices: ", self.indices)
        
        if self.k < self.L.shape[0]: # search swap, p != 2 case     
            # interchange the t^th row and the last row to finalize this swap
            self.indices[t], self.indices[-1] = i, self.indices[t]
            self.W[[t,-1],:] = self.W[[-1,t],:]   # downdate() already zeroes out what was the t^th row/entries
            self.f[[t,-1]] = self.f[[-1,t]]

            # change the Cholesky factor accordingly
            LowRankEnergy.cholesky_add(self.L[:self.k,:self.k], self.G[self.indices[:self.k], i], t)
            LowRankEnergy.cholesky_delete(self.L, self.k)

            if debug and self.verbose:
                print("after downdate, after interchange:")
                print("L: ", np.round(self.L, 1))
                print("W: ", np.round(self.W, 1))
                print("f: ", np.round(self.f, 1))
                print("d: ", np.round(self.d, 1))
                print("indices: ", self.indices)

        else:
            self.update(t, i)

            if debug and self.verbose:
                print("after update:")
                print("L: ", np.round(self.L, 1))
                print("W: ", np.round(self.W, 1))
                print("f: ", np.round(self.f, 1))
                print("d: ", np.round(self.d, 1))
                print("indices: ", self.indices)
        
        

        return 

    def compute_swap_distances(self, idx_to_swap):  # adaptive sampling swap
        if len(self.indices) == 1:  # if we only have a single index in indices, we are going back to sampling proportional to ||x_i|| 
            return np.real(np.diagonal(self.G).copy())
        if self.U is None:
            self.prep_all_downdates(returnU=False)
        return self.U[idx_to_swap,:]

    def compute_eager_swap_values(self, idx): # adaptive search swap
        curr_energy = np.copy(self.energy)
        alg98 = False
        if self.p is not None:
            if np.isclose(self.p, 2) and not self.test:
                alg98 = True
        if alg98: # p = 2 case, (Algorithm 9.8)
            W_rownorm2 = np.linalg.norm(self.W, axis=1)**2.
            ws = self.W[:,idx]
            ws_abs2 = np.abs(ws)**2.
            ys = self.Y[:,idx]
            rs = np.linalg.norm(self.R[:,idx])**2.
            
            denom = self.R[idx,idx] + ws_abs2 / self.f
            skip_mask = np.isclose(denom / self.f, 0.0)   # for values where skip_mask = True, don't compute the swap value (results in non-singular G[S, S] which throws things off)
            denom[skip_mask] = 1.0  # to avoid division by zero warnings

            # vals = U[:,s]
            vals = W_rownorm2/self.f - (rs + W_rownorm2*ws_abs2 / (self.f**2.) + 2.*(ys*ws.conj()).real/self.f) /denom 
            vals[skip_mask] = np.inf  # set these to infinity so they are not chosen
            
            # correct for the different form of the values in the p = 2 case 
            shiftvals = vals + curr_energy**2. 
            vals = np.sqrt(shiftvals)
            if (shiftvals < 0).any():
                print("\tWARNING: p=2 lowrank eager swap values wrong?", vals, curr_energy)

        else: # p != 2, (Algorithm 9.7)
            self.update(t=self.k, i=idx)  # update prototype set at (k+1)th prototype spot with current s 
            self.prep_all_downdates()  # precompute all downdatings including the new prototype at index k
            
            # choose best potential prototype swap, i
            if self.p is None:
                vals = self.U.max(axis=1).flatten()  # already reduced to max entry in prep_all_downdates()
            else:
                # prep_all_downdates() already includes the power p in the computation
                vals = self.U.sum(axis=1)**(1./self.p)

            # make sure the final entry of vals is the current energy
            assert np.isclose(vals[-1], curr_energy)  
            vals = vals[:-1] # since we've passed the above check, we can ignore the last entry, since we know it won't be chosen by .min() operation in sampler 

        return vals  


    @staticmethod
    def cholesky_delete(L, t):
        assert t < L.shape[0] and t >= 0
        L[t,:t] = 0.0 
        L[t,t] = 1.0 
        # only need to do this part if t is not the last index
        if t != L.shape[0] - 1:
            LowRankEnergy.cholesky_update(L[t+1:,t+1:], L[t+1:,t]) # update this part of the Cholesky factor 
            L[t+1:,t] = 0.0
        return
    
    @staticmethod
    def cholesky_update(L, a): 
        """
        Perform a rank-1 Cholesky update of the L matrix with vector a.
        L is updated in place.
        """
        p = L.shape[0]
        for j in range(p):
            r = np.sqrt(abs(L[j,j])**2. + abs(a[j])**2.)
            c = L[j,j] / r
            s = a[j].conj() / r
            L[j:,j] = c * L[j:,j] + s * a[j:]
            a[j:] = (a[j:] - s.conj() * L[j:,j]) / c
        return 
    
    @staticmethod
    def cholesky_add(L, z, t):
        assert t < L.shape[0] and t >= 0
        if t == 0:
            v = z[t]
            c = z[t+1:]
        else:
            a = solve_triangular(L[:t,:t], z[:t], lower=True)
            v = z[t] - np.vdot(a, a).real
            c = z[t+1:] - L[t+1:,:t] @ a
            L[t,:t] = a.conj()
        LowRankEnergy.cholesky_downdate(L[t+1:,t+1:], c/np.sqrt(v))
        L[t,t] = np.sqrt(v)
        L[t+1:,t] = c / np.sqrt(v)
        return

    @staticmethod
    def cholesky_downdate(L, a):
        """
        Perform a rank-1 Cholesky downdate of the L matrix with vector a.
        L is changed in place.
        """
        p = L.shape[0]
        for j in range(p):
            if abs(L[j,j])**2. <= abs(a[j])**2.:
                raise ValueError("Cholesky downdate will have complex diagonal... something wrong.")
            r = np.sqrt(abs(L[j,j])**2. - abs(a[j])**2.)
            c = L[j,j] / r
            s = a[j].conj() / r
            L[j:,j] = c * L[j:,j] - s * a[j:]
            a[j:] = (a[j:] - s.conj() * L[j:,j]) / c
        return 





class ConicHullEnergy(EnergyClass):
    def __init__(self, X, p=2, n_jobs=4, verbose=False):
        super().__init__(X, p=p)
        assert self.X.min() >= -1e-13 # ensure non-negativity
        self.sparse_flag = sps.issparse(self.X)
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
        self.W = np.zeros((self.k, self.X.shape[0]))
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
        
        return

    def compute_swap_distances(self, idx_to_swap):
        if len(self.indices) == 1:  # if we only have a single index in indices, we are going back to sampling via ||x_i||
            dists = np.linalg.norm(self.X, ord=2, axis=0).flatten()
        else:
            dists, _ = self.nnls_OGM_gram(idx_to_swap=idx_to_swap, returnH=False)
        if self.p is None:
            return 1.0*(dists == np.max(dists))   # mask where dists is largest
        return (dists/dists.max())**(self.p)

    def swap(self, t, i, debug=False):
        assert (t < len(self.indices))  and (t >= 0)
        if i in self.indices:
            print(f"Warning: {i} already in self.indices -- not a valid swap. Skipping...")
            return 
        self.indices[t] = i 
        self.G_S[t,:] = self.X.T @ self.X[:,i].flatten()
        dists, H = self.nnls_OGM_gram(returnH=True) 
        self.H = H   # assuming swap is only done with len(self.indices) = self.k
        self.dists = dists 
        self.compute_energy()
        
    
    def search_distances(self, candidates):
        if self.verbose:
            iterator = tqdm(candidates, total=len(candidates))
            iterator.set_description(f"Computing conic hull search values... len(self.indices) = {len(self.indices)}")
        else:
            iterator = candidates
        
        if self.n_jobs is not None:
            with parallel_backend("loky", inner_max_num_threads=1):
                outs = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.nnls_OGM_gram)(search_ind=c, returnH=False) for c in iterator)
                search_dists = [out[0] for out in outs]
        else:
            search_dists = [self.nnls_OGM_gram(search_ind=c, returnH=False)[0] for c in iterator]
        
        
        
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
        dist_vals = np.sqrt(dist_vals )    # we consider Euclidean distances 
        # if verbose:
        #     return {"dist_vals":dist_vals, "iters":i, "eps":eps, "eps0":eps0, "EPS":EPS }, H

        if verbose:
            with open(f"./results/conic_hull_energy_log_{self.n}.txt", "a") as f:
                f.write(f"{len(self.indices)},{self.k},{i}\n")
        if returnH:
            return dist_vals, H 
        
        return dist_vals, None





class ConvexHullEnergy(EnergyClass):
    def __init__(self, X, p=2, n_jobs=4, verbose=False):
        super().__init__(X, p=p)
        self.X = X
        # center the data (usual practice in archetypal analysis)
        self.X -= self.X.mean(axis=1, keepdims=True)
        self.G = self.X.T @ self.X
        self.Gdiag = np.diagonal(self.G).flatten()
        self.dists = np.sqrt(self.Gdiag).flatten()  # initial distances are just ||x_i||
        self.compute_energy()
        self.use_previous = True
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.type = "convex"
        self.sst = float(np.sum(self.Gdiag))  # Total sum of squares: ||X||_F^2
        

    def set_k(self, k):
        super().set_k(k)
        self.W = np.zeros((self.X.shape[0], self.k))
        self.H = np.zeros((self.k, self.n))
        self.G_S = np.zeros((self.k, self.n)) 
        return 
    
        
    def add(self, i):
        if self.k is None:
            raise NotImplementedError("Iterative allocation of memory for ConicHullEnergy objects not yet implemented. Must set desired k with ConicHullEnergy.set_k(k)")
        self.W[:,len(self.indices)] = self.X[:,i].flatten()
        self.G_S[len(self.indices),:] = self.G[:,i].flatten()
        self.indices.append(i)
        dists, H = self.nnls_gram(returnH=True)
        self.H[:len(self.indices),:] = H 
        self.dists = dists 
        self.compute_energy()
        
        return

    def compute_swap_distances(self, idx_to_swap):
        if len(self.indices) == 1:  # if we only have a single index in indices, we are going back to sampling via ||x_i||
            dists = np.sqrt(self.Gdiag).flatten()
        else:
            dists, _ = self.nnls_gram(idx_to_swap=idx_to_swap, returnH=False)
        if self.p is None:
            return 1.0*(dists == np.max(dists))   # mask where dists is largest
        return (dists/dists.max())**(self.p)

    def swap(self, t, i, debug=False):
        assert (t < len(self.indices))  and (t >= 0)
        if i in self.indices:
            print(f"Warning: {i} already in self.indices -- not a valid swap. Skipping...")
            return 
        self.indices[t] = i 
        self.G_S[t,:] = self.G[:,i].flatten()
        dists, H = self.nnls_gram(returnH=True) 
        self.H = H   # assuming swap is only done with len(self.indices) = self.k
        self.dists = dists 
        self.compute_energy()
        
    
    def search_distances(self, candidates):
        if self.verbose:
            iterator = tqdm(candidates, total=len(candidates))
            iterator.set_description(f"Computing convex hull search values... len(self.indices) = {len(self.indices)}")
        else:
            iterator = candidates
        
        if self.n_jobs is not None:
            with parallel_backend("loky", inner_max_num_threads=1):
                outs = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.nnls_gram)(search_ind=c, returnH=False) for c in iterator)
                search_dists = [out[0] for out in outs]
        else:
            search_dists = [self.nnls_gram(search_ind=c, returnH=False)[0] for c in iterator]
        
        
        
        return np.array(search_dists) 

    def compute_eager_swap_values(self, idx):
        if idx in self.indices:
            return np.ones(len(self.indices))*self.energy*(1.0000001)
        r = np.hstack([self.nnls_gram(search_ind=idx, idx_to_swap=j, returnH=False)[0].reshape(-1, 1) \
                      for j in range(len(self.indices))])
        if self.p is None:
            r = np.max(r, axis=0)
        else:
            r = np.linalg.norm(r, axis=0, ord=self.p)
        return r
        
    

    def nnls_gram(self, search_ind=None, idx_to_swap=None, delta=1e-6, seed=42, maxiter=500, returnH=True, verbose=False, muH_init=1.0, inner_iters=10):
        """
        Fit convex weights H for fixed prototypes W = X[:, idx],
        using only the Gram matrix G = X^T X.

        Solves:
            min_H ||X - W H||_F^2
            s.t. H >= 0, sum_k H_{k,j} = 1 for all j

        Parameters
        ----------
        search_ind : int   (0 <= search_ind < n)
            Index of candidate prototype to include 
        idx_to_swap : int  (0 <= idx_to_swap < k)
            Index of current prototype to remove 
        delta : float
            Relative squared distances convergence tolerance
        seed : int 
            Random state seed for reproducibility
        maxiter : int
            Maximum outer iterations
        returnH : bool 
            Return the convex weights H after the computation
        verbose : bool
            Print convergence diagnostics
        muH_init : float
            Initial step size for projected gradient
        inner_iters : int
            Number of projected-gradient steps per outer iteration
        

        Returns
        -------
        dist_vals : float
            Root sum of squared reconstruction error
        H : (k, n) ndarray
            Convex weights (columns sum to 1)
        """
        rand_state = np.random.RandomState(seed)

        S_ind_all = self.indices[:]

        if idx_to_swap is not None:
            if search_ind is not None:
                S_ind_all[idx_to_swap] = search_ind 
            else:
                S_ind_all = S_ind_all[:idx_to_swap] + S_ind_all[idx_to_swap+1:]
        else:
            if search_ind is not None:
                S_ind_all = S_ind_all + [search_ind]

        # --- Gram submatrices ---
        WtW = self.G[np.ix_(S_ind_all, S_ind_all)]                 # (k, k)
        WtX = self.G[S_ind_all,:]        # (k, n)

        # --- Initialize H ---
        H = -np.log(rand_state.random(WtX.shape))
        H = H / (np.sum(H, axis=0, keepdims=True) + 1e-12)

        # Initial d (squared distances)
        HHt = H @ H.T
        d = self.sst - 2.0 * np.sum(WtX * H) + np.sum(WtW * HHt)

        muH = float(muH_init)
        dd = np.inf
        it = 0
        e = np.ones((len(S_ind_all), 1))

        # --- Main loop ---
        while (abs(dd) >= delta * abs(d)) and it < maxiter:
            it += 1
            d_old = d

            for _ in range(inner_iters):
                d_inner_old = d

                # Gradient: g = (W^T W H - W^T X) / (SST / n)
                g = (WtW @ H - WtX) / (self.sst / float(self.n))

                # Project gradient onto simplex tangent space
                g = g - e @ np.sum(g * H, axis=0, keepdims=True)

                Hold = H
                while True:
                    H = Hold - muH * g
                    H = ConvexHullEnergy._project_to_simplex_columns(H)

                    HHt = H @ H.T
                    d = self.sst - 2.0 * np.sum(WtX * H) + np.sum(WtW * HHt)
                    
                    if d <= d_inner_old * (1.0 + 1e-9):
                        muH *= 1.2
                        break
                    muH /= 2.0

            dd = d_old - d

            if verbose:
                print(f"iter {it:4d} | d {d: .4e} | ",  fr"Delta-rel {dd/abs(d): .2e}")

        dist_vals = np.clip(np.diagonal(self.G) - 2.*(WtX * H).sum(axis=0) + ((WtW @ H) * H).sum(axis=0), 0.0, None)
        assert np.isclose(dist_vals.sum(), d)
        dist_vals = np.sqrt(dist_vals)    # revert to Euclidean distances
        if returnH:
            return dist_vals, H 
        
        return dist_vals, None
    
    @staticmethod
    def _project_to_simplex_columns(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Project columns onto the probability simplex via
        nonnegativity + column normalization.
        """
        A = np.maximum(A, 0.0)
        return A / (np.sum(A, axis=0, keepdims=True) + eps)