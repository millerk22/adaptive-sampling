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
    
    def build(self, k):
        return 
    
    def swap(self, num_swaps):
        return 
    

class AdaptiveSampling(Sampler):
    def __init__(self, Energy_, seed=42, report_timing=False):
        super().__init__(Energy_, seed=seed, report_timing=report_timing)
        
    
    def build(self, k):
        for i in range(k):
            if self.report_timing:
                tic = perf_counter()
            if self.Energy.p is not None:
                q = self.Energy.dists / self.Energy.dists.sum()
                idx = self.random_state.choice(range(self.Energy.n), p=q)
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

    def swap(self, num_swaps):
        return 
    

class AdaptiveSearch(Sampler):
    def __init__(self, Energy_, seed=42, report_timing=False):
        super().__init__(Energy_, seed=seed, report_timing=report_timing)
    
    def build(self, k):
        for i in range(k):
            if self.report_timing:
                tic = perf_counter()
            # compute the look-ahead energies for all the non-included points
            q_look_aheads_p = self.Energy.look_ahead() # includes the power p in the computation in self.Energy

            # arbitratily choose the minimizer of the look_aheads 
            min_idxs = np.where(q_look_aheads_p == np.min(q_look_aheads_p))[0]
            idx = self.random_state.choice(min_idxs)

            if self.report_timing:
                toc = perf_counter()
                self.times.append(toc-tic)

            # update distances and add point to index set
            self.Energy.add(idx)
        return 

    def swap(self, num_swaps):
        return 
    






def perform_swaps_for_all(X, kstart, indices, swap_method,report_timing=False, verbose=True):
    assert kstart <= len(indices)
    assert swap_method.split("-")[0] in ["greedyla", "p", "greedy"]

    print(f"------ Performing swap moves on each k = [{kstart}, {len(indices)}] ---------")
    energy = []
    indices_k = []
    times_all = []
    for k in range(len(indices)):
        if k < kstart:
            energy.append(-1.0)
            indices_k.append(indices)
            times_all.append([])
            continue
        print(f"k = {k},   {k-kstart+1}/{len(indices)-kstart}")
        Energy = ConvexHullEnergy(X, k)
        Energy.init_set(indices[:k])
        times = []

        for u in range(max(k**2 //2, 5)):
            if report_timing:
                t0 = perf_counter()
            swap_flag = Energy.swap_move(swap_method, j_adap=u % k, verbose=verbose)
            if report_timing:
                t1 = perf_counter()
                times.append(t1-t0)
            # termination check for the different cases
            if swap_method == "greedy":
                if not swap_flag:
                    no_change_count += 1
                else:
                    no_change_count = 0
                if no_change_count == k:
                    break
            else: # adaptive swaps with p < \infty (not "greedy") always return swap_flag = True, so this just handles the greedyla termination condition
                if not swap_flag:
                    break

        energy.append(Energy.energy)
        indices_k.append(Energy.indices)
        times_all.append(times)

    return energy, indices_k, times_all

