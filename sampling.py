import numpy as np
from sklearn.utils.extmath import stable_cumsum
from functools import partial
from energies import *
from time import perf_counter



class Sampler(object):
    def __init__(self, Energy_, seed=42, report_timing=False):
        self.Energy = Energy_ 
        self.seed = 42
        self.report_timing = report_timing
        self.random_state = np.random.RandomState(self.seed)
        self.times = []
    
    def build_phase(self, k):
        return 
    
    def swap_phase(self, num_swaps=None):
        return 

# CHECK THAT ADAPTIVE SAMPLING SWAP is DONE CORRECT

class AdaptiveSampling(Sampler):
    def __init__(self, Energy_, seed=42, report_timing=False):
        super().__init__(Energy_, seed=seed, report_timing=report_timing)
        
    
    def build_phase(self, k):
        for i in range(k):
            if self.report_timing:
                tic = perf_counter()
            if self.Energy.p is not None:
                q = self.Energy.dists**(self.Energy.p)
                idx = self.random_state.choice(range(self.Energy.n), p=q/q.sum())
            else:
                # adaptive sampling with p = infty
                max_idxs = np.where(self.Energy.dists == self.Energy.energy)[0]
                idx = self.random_state.choice(max_idxs)

            if self.report_timing:
                toc = perf_counter()
                self.times.append(toc-tic)

            # update distances and add point to index set
            self.Energy.add(idx)
        return 

    def swap_phase(self, num_swaps):
        k = len(self.Energy.indices)
        for u in range(num_swaps):
            t = u % k
            q_probs_wo_st = self.Energy.compute_search_values(idx_to_swap=t)
            q_probs_wo_st[inds_wo_st] = 0.0       # don't want to give any probability to those points that are already in the current indices set
            q_probs_wo_st = q_probs_wo_st**(self.Energy.p)
            

            # sample the swap move point
            if self.Energy.p is None:
                max_idxs = np.where(q_probs_wo_st == np.max(q_probs_wo_st))[0]
                idx = self.random_state.choice(max_idxs)
            else:
                idx = self.random_state.choice(range(self.Energy.n), p=q_probs_wo_st/q_probs_wo_st.sum())
            
            # update the Energy object according to this swap move
            self.Energy.swap(t, idx) 

        return 
    

class AdaptiveSearch(Sampler):
    def __init__(self, Energy_, seed=42, report_timing=False):
        super().__init__(Energy_, seed=seed, report_timing=report_timing)
    
    def build_phase(self, k):
        for i in range(k):
            if self.report_timing:
                tic = perf_counter()
            # compute the search energies for all the non-included points
            q_search_p = self.Energy.compute_search_values() # includes the power p in the computation in self.Energy

            # arbitratily choose the minimizer of the search
            min_idxs = np.where(q_search_p == np.min(q_search_p))[0]
            idx = self.random_state.choice(min_idxs)

            if self.report_timing:
                toc = perf_counter()
                self.times.append(toc-tic)

            # update distances and add point to index set
            self.Energy.add(idx)
        return 

    def swap_phase(self, num_swaps=None):
        k = len(self.Energy.indices)
        n = self.Energy.n
        swap_flag = True 

        while swap_flag:
            
            C = np.zeros((n, k))
            for t in range(k):
                C[:,t] = self.Energy.compute_search_values(idx_to_swap=t) # candidates will be the unselected inds, since we consider all entries
                                                                    # of C that correspond to the current indices (S) will just have the current energy value.
            
            assert (C[np.ix_(self.Energy.indices, self.Energy.indices)] == self.Energy.energy).all()
            
            # find swap move in argmin_{idx,t} C(idx,t). arbitrarily break ties
            idx_poss, t_poss = np.where(C == np.min(C))
            jstar = self.random_state.choice(len(idx_poss))
            idx, t = idx_poss[jstar], t_poss[jstar]

            # if the minimum is the current energy, then no swap results in a better energy, so stop swapping
            if idx in self.Energy.indices:
                break 
            else:
                self.Energy.swap(t, idx)
            
        return 
