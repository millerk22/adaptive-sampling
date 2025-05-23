import numpy as np
from energies import *
from time import perf_counter


class AdaptiveSampler(object):
    def __init__(self, Energy_, seed=42, report_timing=False):
        self.Energy = Energy_ 
        self.seed = seed
        self.report_timing = report_timing
        self.random_state = np.random.RandomState(self.seed)
        self.times = []
    
    def build_phase(self, k, method="sampling"):
        assert method in ["sampling", "search"]
        assert len(self.Energy.indices) == 0   # ensure we are starting with an "empty" indices set
        self.Energy.set_k(k)                   # allocate memory in the Energy object for the k points to be chosen

        for i in range(k):
            if self.report_timing:
                tic = perf_counter()
            
            if method == "search": # adaptive-search build
                # compute the search energies for all the non-included points
                q_search_p = self.Energy.compute_search_values() # includes the power p in the computation in self.Energy

                # arbitratily choose the minimizer of the search
                min_idxs = np.where(q_search_p == np.min(q_search_p))[0]
                idx = self.random_state.choice(min_idxs)
            else: # adaptive-sampling build
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
    
    def swap_phase(self, method="search", max_swaps=None):
        assert method in ["search", "sampling"]
        k = len(self.Energy.indices)
        n = self.Energy.n
        if max_swaps is None:
            max_swaps = k**2

        if method == "search": # adaptive-search swap
            swap_flag = True 
            count = 0
            while swap_flag:
                if self.report_timing:
                    tic = perf_counter()
                C = np.zeros((n, k))
                for t in range(k):
                    C[:,t] = self.Energy.compute_search_values(idx_to_swap=t) # candidates will be the unselected inds, since we consider all entries
                                                                        # of C that correspond to the current indices (S) will just have the current energy value.
                
                assert (C[self.Energy.indices, :] == self.Energy.energy).all()
                
                # find swap move in argmin_{idx,t} C(idx,t). arbitrarily break ties
                idx_poss, t_poss = np.where(C == np.min(C))
                jstar = self.random_state.choice(len(idx_poss))
                idx, t = idx_poss[jstar], t_poss[jstar]

                if self.report_timing:
                    toc = perf_counter()
                    self.times.append(toc-tic)

                # if the minimum is the current energy, then no swap results in a better energy, so stop swapping
                if idx in self.Energy.indices:
                    break 
                else:
                    self.Energy.swap(t, idx)
                
                count += 1
                if count > max_swaps:
                    break
                
        else:
            # adaptive-sampling swap 
            check_change_flag = False
            if self.Energy.p is None:
                no_change_count = 0
                check_change_flag = True

            for u in range(max_swaps):
                if self.report_timing:
                    tic = perf_counter()
                t = u % k
                if check_change_flag:
                    old_idx = self.Energy.indices[t]

                q_probs_wo_st = self.Energy.compute_search_values(idx_to_swap=t)
                inds_wo_st = self.Energy.indices[:t] + self.Energy.indices[t+1:]
                q_probs_wo_st[inds_wo_st] = 0.0       # don't want to give any probability to those points that are already in the current indices set
                
                # sample the swap move point
                if self.Energy.p is None:
                    max_idxs = np.where(q_probs_wo_st == np.max(q_probs_wo_st))[0]
                    idx = self.random_state.choice(max_idxs)
                else:
                    q_probs_wo_st = q_probs_wo_st**(self.Energy.p)
                    idx = self.random_state.choice(range(self.Energy.n), p=q_probs_wo_st/q_probs_wo_st.sum())
                
                if self.report_timing:
                    toc = perf_counter()
                    self.times.append(toc-tic)
                
                # update the Energy object according to this swap move
                self.Energy.swap(t, idx) 
                
                # check if the index set has not changed for k consecutive iterations
                if check_change_flag:
                    if idx == old_idx:
                        no_change_count += 1 
                    else:
                        no_change_count = 0
                    old_idx = idx   
                    if no_change_count == k:
                        break
        
        return 

