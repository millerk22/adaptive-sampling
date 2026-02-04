import numpy as np
from energies import *
from time import perf_counter
from util import *


class AdaptiveSampler(object):
    def __init__(self, Energy_, seed=42, record=False):
        self.Energy = Energy_ 
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
        self.build_times = []
        self.swap_times = []
        self.build_values = []
        self.swap_values = []
        self.swap_stag = []
        self.record = record
    
    def record_build(self, t, e):
        self.build_times.append(t)
        self.build_values.append(e)
        return 
    
    def record_swap(self, t, e, w):
        self.swap_times.append(t)
        self.swap_values.append(e)
        self.swap_stag.append(w)
        return 

    def build_phase(self, k, method="sampling"):
        assert method in ["sampling", "search", "uniform"]
        assert len(self.Energy.indices) == 0   # ensure we are starting with an "empty" indices set

        if method == "uniform":
            # have uniform sampling implemented here as well for ease in the run_experiment function in util.py
            indices = list(self.random_state.choice(self.Energy.n, k, replace=False))
            build_values = self.Energy.init_set(indices, return_values=True)[1:]
            self.build_values = build_values
            self.build_times = k*[None]
        else:
            self.Energy.set_k(k)          # allocate memory in the Energy object for the k points to be chosen
            for i in range(k):
                tic = perf_counter()
                
                if method == "search": # adaptive-search build
                    # compute the search energies for all the non-included points
                    q_search_p = self.Energy.compute_search_values() # includes the power p in the computation in self.Energy

                    # arbitratily choose the minimizer of the search
                    min_idxs = np.where(q_search_p == np.min(q_search_p))[0]
                    idx = self.random_state.choice(min_idxs)

                else: # adaptive-sampling build
                    if self.Energy.p is not None:
                        q = (self.Energy.dists/self.Energy.dists.max())**(self.Energy.p)
                        #q = self.Energy.dists**(self.Energy.p)
                        idx = self.random_state.choice(range(self.Energy.n), p=q/q.sum())
                    else:
                        # adaptive sampling with p = infty
                        max_idxs = np.where(self.Energy.dists == self.Energy.energy)[0]
                        idx = self.random_state.choice(max_idxs)

                # update distances and add point to index set
                self.Energy.add(idx)

                if self.record:
                    toc = perf_counter()
                    self.record_build(toc-tic, self.Energy.energy)

            
        return 
    
    def swap_phase(self, method="sampling", debug=False, max_swaps=-1, test=False):
        assert method in ["search", "sampling"]
        assert type(max_swaps) == int
        k = len(self.Energy.indices)
        n = self.Energy.n
        self.Energy.prep_for_swaps(method)

        if method == "search":
            s, w = 0, 0
            num_swaps = 0
            if self.record:
                tic = perf_counter()
            while w < n:
                if debug:
                    print(s, self.Energy.indices, self.Energy.energy)
                    print("\t", [np.allclose(self.Energy.X[:,s], self.Energy.X[:,i]) for i in self.Energy.indices])
                if s in self.Energy.indices:
                    s = (s + 1) % n 
                    w += 1
                    continue 
                
                curr_energy = np.copy(self.Energy.energy)

                # construct the eager swap vector, r_i vectors (with prototype indices {S/s_t \cup i} for each s_t in S currently)
                vals = self.Energy.compute_eager_swap_values(idx=s) 
                assert vals.size == k

                swap = vals.min() < curr_energy
                
                if debug:
                    new_energy = vals.min()
                
                # check if we perform a swap
                if swap: 
                    t_poss = np.where(np.isclose(vals, vals.min()))[0]
                    t = t_poss[self.random_state.choice(len(t_poss))]
                    s_old = self.Energy.indices[:][t]

                    self.Energy.swap(t, s, debug=debug)
                    num_swaps += 1

                    if debug:
                        # check if R, Y, W, L are still correct after some swaps 
                        print("Checking components after ", num_swaps , " swaps...")
                        check_components(self.Energy, nround=5, checkR=self.Energy.p==2, verbose=False)

                        if not np.isclose(new_energy, self.Energy.energy):
                            print(f"Warning (Search Swap): Updated energy is {self.Energy.energy},\n\tthought it was going to be {new_energy}")
                            print(f"\tt={t}, s={s}, previous s_t = {s_old}")
                            energy_ = LowRankEnergy(self.Energy.X, p=self.Energy.p)
                            energy_.init_set(self.Energy.indices[:t] + [s] + self.Energy.indices[t+1:])
                            print(f"\tRecomputed energy from scratch is {energy_.energy}")

                    # record the time and energy for the swap move
                    if self.record:
                        toc = perf_counter()
                        self.record_swap(toc-tic, self.Energy.energy, w) # record info 
                        tic = perf_counter()
                    
                    
                    # in the case that a max number of swaps is specified, check here if need to terminate
                    #     (really only used with ConicEnergy because is costly)
                    
                    if num_swaps == max_swaps:
                        print(f"-----Not necessarily converged, but reached specified maximum number of swaps ({max_swaps})-----")
                        print("\tTerminating...")
                        break

                    # reset stagnation counter
                    w = 0
 
                else:
                    if self.Energy.type == "lowrank": 
                        if self.Energy.p is None:
                            self.Energy.downdate(k)
                        else:
                            if not np.isclose(self.Energy.p, 2):# swap was unsuccessful, so revert the temporary update
                                self.Energy.downdate(k)
                            elif np.isclose(self.Energy.p, 2) and self.Energy.test:
                                self.Energy.downdate(k)
                    w += 1

                s = (s + 1) % n

            if self.record: # record the time at the end for completeness
                toc = perf_counter()
                self.record_swap(toc-tic, self.Energy.energy, w) 

        elif method == "sampling":   # adaptive sampling swap with swap forcing (Currently Alg 3.4)    
            # initialize counters, force swap prob vector, and best prototype trackers
            t, w, u = 0, 0, 0
            p = np.zeros(k)
            best_energy, best_inds = self.Energy.energy, self.Energy.indices[:]  # will track the best found energy and indices through adaptive swap moves
            
            if self.record:
                tic = perf_counter()
            
            # Complete at most 2*k successful swaps
            while u < 2*k:
                if self.Energy.type == "lowrank":
                    U = self.Energy.prep_all_downdates(returnU=True) # precompute all downdatings for low-rank energy. 
                                                        # (includes the power p in the computation)
                    denom = U.sum(axis=1)
                    p = U[list(range(k)),self.Energy.indices] / denom
                    p = np.roll(p, -t)  # roll so that p[0] corresponds to current t index


                    # if stagnated, then terminate. With low-rank, we are always forcing a swap if possible. 
                    # This is the case when no swap is possible.
                    if np.isclose(np.prod(p), 1.0):
                        break 

                    # always force a swap in low-rank case 
                    probs = np.concatenate(([1], np.cumprod(p)[:-1])) * (1. - p) / (1. - np.prod(p))  # probabilities for selecting which prototype to force a swap
                    j = self.random_state.choice(range(k), p=probs)   
                    t = (t + j) % k  # no +1 because p[0] corresponds to prob of current t index
                    force_swap_probs = U[t,:]/(denom[t] - U[t,self.Energy.indices[t]])
                    force_swap_probs[self.Energy.indices[t]] = 0.0
                    assert np.isclose(force_swap_probs.sum(), 1.0) # check valid probability distribution
                    s_prime = self.random_state.choice(range(self.Energy.n), p=force_swap_probs) # sample the forced swap prototype

                else: # Algorithm 3.5
                    # Step 1: Compute probability of trivial swap
                    q_probs_wo_st = self.Energy.compute_swap_distances(idx_to_swap=t)  # includes the power p from the Energy object computation
                    q_probs_den = q_probs_wo_st.sum()
                    p[w] = q_probs_wo_st[self.Energy.indices[t]] /q_probs_den # trivial swap prob
                    

                    # Step 2: Does a swap happen?
                    if self.random_state.rand() > p[w]:
                        q_probs_den -= q_probs_wo_st[self.Energy.indices[t]] # subtract off the trivial swap component
                        q_probs_wo_st[self.Energy.indices[t]] = 0.0          # ignore s_t
                        non_trivial_swap_probs = q_probs_wo_st / q_probs_den
                        assert np.isclose(non_trivial_swap_probs.sum(), 1.0) # check valid probability distribution
                        s_prime = self.random_state.choice(range(self.Energy.n), p=non_trivial_swap_probs)
                    else:
                        s_prime = self.Energy.indices[t]
                        w += 1
                    
                    # Step 3: If stagnated, then terminate (p = \inf case)
                    if (s_prime == self.Energy.indices[t]) and (w == k) and (self.Energy.p is None):
                        break

                    # Step 4: If stagnated, then force a swap (p < \inf case)
                    if (s_prime == self.Energy.indices[t]) and (w == k) and (self.Energy.p is not None):
                        print("FORCED SWAP, w = ", w)
                        probs = np.concatenate(([1], np.cumprod(p)[:-1])) * (1. - p) / (1. - np.prod(p))  # probabilities for selecting which prototype to force a swap
                        j = self.random_state.choice(range(k), p=probs)   
                        t = (t + j + 1) % k # +1 because p[0] corresponds to prob of next index after current t
                        q_probs_wo_st = self.Energy.compute_swap_distances(idx_to_swap=t)  # with new index, compute cost vector probs
                        q_probs_den = q_probs_wo_st.sum()                    
                        force_swap_probs = q_probs_wo_st/(q_probs_den - q_probs_wo_st[self.Energy.indices[t]])  
                        force_swap_probs[self.Energy.indices[t]] = 0.0
                        assert np.isclose(force_swap_probs.sum(), 1.0) # check valid probability distribution
                        s_prime = self.random_state.choice(range(self.Energy.n), p=force_swap_probs) # sample the forced swap prototype
                    
                # Step 4: perform the swap if swap was found
                if s_prime != self.Energy.indices[t]:
                    self.Energy.swap(t, s_prime)
                    u += 1                   # increment swap counter
                    # track best energy found so far
                    if self.Energy.energy < best_energy:
                        best_energy = self.Energy.energy
                        best_inds = self.Energy.indices[:]

                    # record the time, energy, and stagnation counter for this swap
                    if self.record:
                        toc = perf_counter()
                        self.record_swap(toc-tic, self.Energy.energy, w) # record info 
                        tic = perf_counter()
                    w = 0                       # reset stagnation counter

                t = (t + 1) % k    # increment index counter
            
            if self.record: # record the time at the end for completeness
                toc = perf_counter()
                self.record_swap(toc-tic, self.Energy.energy, w)

            # At the end of the swap phase, set the indices to the best found
            #     - first re-initialize the Energy object to clear out the current indices
            if debug:
                print("Best energy found during sampling swap:", best_energy)
                print("Energy at end of sampling swap:", self.Energy.energy)
                print("Re-initializing Energy object to best found indices...")
            self.Energy.__init__(self.Energy.X, p=self.Energy.p)  # re-initialize Energy object
            self.Energy.init_set(best_inds)
            
            
            if not np.isclose(self.Energy.energy, best_energy): # check that we have properly reset the Energy object
                print("Warning...: ", best_energy, self.Energy.energy, "re-initialized energy is not the same?")
            

        else:
            raise NotImplementedError(f"Method {method} not recognized for swap phase.")
        
        # clean up any temporary variables used in respective Energy objects
        self.Energy.end_swaps()

        return 

