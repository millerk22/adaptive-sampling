import numpy as np
from energies import *
from time import perf_counter


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
                        q = self.Energy.dists**(self.Energy.p)
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
    
    def swap_phase(self, method="sampling", debug=False, max_swaps=-1):
        assert method in ["search", "sampling"]
        assert type(max_swaps) == int
        k = len(self.Energy.indices)
        n = self.Energy.n
        self.Energy.prep_for_swaps(method)
        
        if method == "search" and self.Energy.type == "lowrank":
            s, w = 0, 0
            if self.record:
                tic = perf_counter()
            while w < n:
                if s in self.Energy.indices:
                    s = (s + 1) % n 
                    w += 1
                    continue 
                
                curr_energy = np.copy(self.Energy.energy)
                self.Energy.update(t=k, i=s)  # update prototype set at (k+1)th prototype spot with current s 
                U = self.Energy.prep_all_downdates(returnU=True)  # precompute all downdatings including the new prototype at index k
                
                # choose best potential prototype swap, i
                if self.Energy.p is None:
                    vals = U.flatten()  # already reduced to max entry in prep_all_downdates()
                else:
                    # prep_all_downdates() already includes the power p in the computation
                    vals = U.sum(axis=1)**(1./self.Energy.p)            
                t_poss = np.where(np.isclose(vals,np.min(vals)))[0]
                t = t_poss[self.random_state.choice(len(t_poss))]
                
                # check that the last entry corresponds to current energy 
                assert np.isclose(vals[-1], curr_energy)  
                
                if vals[t] < curr_energy and t != k:  # did this swap improve energy?
                    if debug:
                        print(f"before downdate to swap at {t}, {self.Energy.indices[t]} and {self.Energy.indices[-1]} :")
                        print("L: ", np.round(self.Energy.L, 1))
                        print("W: ", np.round(self.Energy.W, 1))
                        print("f: ", np.round(self.Energy.f, 1))
                        print("d: ", np.round(self.Energy.d, 1))
                        print("indices: ", self.Energy.indices)
                    self.Energy.downdate(t)

                    if debug:
                        print("after downdate, before interchange:")
                        print("L: ", np.round(self.Energy.L, 1))
                        print("W: ", np.round(self.Energy.W, 1))
                        print("f: ", np.round(self.Energy.f, 1))
                        print("d: ", np.round(self.Energy.d, 1))
                        print("indices: ", self.Energy.indices)
                    # interchange the t^th row and the last row to finalize this swap
                    self.Energy.indices[t], self.Energy.indices[-1] = s, self.Energy.indices[t]
                    self.Energy.W[[t,-1],:] = self.Energy.W[[-1,t],:]   # downdate() already zeroes out what was the t^th row/entries
                    self.Energy.f[[t,-1]] = self.Energy.f[[-1,t]]

                    # change the Cholesky factor accordingly
                    LowRankEnergy.cholesky_add(self.Energy.L[:k,:k], self.Energy.G[self.Energy.indices[:k], s], t)
                    LowRankEnergy.cholesky_delete(self.Energy.L, k)

                    # record the time and energy for the swap move
                    if self.record:
                        toc = perf_counter()
                        self.record_swap(toc-tic, self.Energy.energy, w) # record info 
                        tic = perf_counter()

                    w = 0

                    if debug:
                        print("after downdate, after interchange:")
                        print("L: ", np.round(self.Energy.L, 1))
                        print("W: ", np.round(self.Energy.W, 1))
                        print("f: ", np.round(self.Energy.f, 1))
                        print("d: ", np.round(self.Energy.d, 1))
                        print("indices: ", self.Energy.indices)
                else:
                    # swap was unsuccessful, so revert the temporary update
                    self.Energy.downdate(k)
                    w += 1

                s = (s + 1) % n

            if self.record:
                toc = perf_counter()
                self.record_swap(toc-tic, self.Energy.energy, w) 

        elif method == "search": # adaptive-search swap
            i, w = 0, 0
            num_swaps = 0
            if self.record:
                tic = perf_counter()
            while w < n:
                if debug:
                    print(i, w, n, self.Energy.indices)
                # skip x_i if is already in the prototype set
                if i in self.Energy.indices:
                    i += 1
                    i = i % n
                    continue
                
                # construct the eager swap vector, r_i vectors (with prototype indices {S/s_t \cup i} for each s_t in S currently)
                r = self.Energy.compute_eager_swap_values(idx=i) 
                assert r.size == k

                # select one of the minimizers
                j_poss = np.where(r == np.min(r))[0]
                jstar = j_poss[self.random_state.choice(len(j_poss))]

                
                # if minimum swap value improves the current energy, then perform the swap and update counter
                if r[jstar] < self.Energy.energy:
                    if self.Energy.indices[jstar] != i: # make sure we are actually making a swap
                        old_idx = self.Energy.indices[:][jstar]
                        self.Energy.swap(jstar, i)
                        if not np.isclose(self.Energy.energy, r[jstar]):
                            print("Warning: Adaptive Search Swap")
                            print("\t", jstar, old_idx, i, "not close", len(j_poss), len(self.Energy.indices), self.Energy.indices)
                            print("\t", r, r[jstar], self.Energy.energy)
                            
                        # record the time, energy, and stagnation counter for the swap move
                        if self.record:
                            toc = perf_counter()
                            self.record_swap(toc-tic, self.Energy.energy, w) # record info 
                            tic = perf_counter()

                        # in the case that a max number of swaps is specified, check here if need to terminate
                        #     (really only used with ConicEnergy because is costly)
                        num_swaps += 1
                        if num_swaps == max_swaps:
                            print(f"-----Not necessarily converged, but reached specified maximum number of swaps ({max_swaps})-----")
                            print("\tTerminating...")
                            break

                        w = 0 # reset the stagnation counter
                        
                    else:
                        w += 1
                else:
                    w += 1
                
                # update index
                i += 1 
                i = i % n
            
            if self.record:
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
                    w = 0                       # reset stagnation counter
                    # track best energy found so far
                    if self.Energy.energy < best_energy:
                        best_energy = self.Energy.energy
                        best_inds = self.Energy.indices[:]

                    # record the time, energy, and stagnation counter for this swap
                    if self.record:
                        toc = perf_counter()
                        self.record_swap(toc-tic, self.Energy.energy, w) # record info 
                        tic = perf_counter()

                t = (t + 1) % k    # increment index counter

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

