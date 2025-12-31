import numpy as np
from energies import *
from time import perf_counter


class AdaptiveSampler(object):
    def __init__(self, Energy_, seed=42, report_timing=False):
        self.Energy = Energy_ 
        self.seed = seed
        self.report_timing = report_timing
        self.random_state = np.random.RandomState(self.seed)
        self.build_times = []
        self.swap_times = []

    def build_phase(self, k, method="sampling"):
        assert method in ["sampling", "search", "searchwithinituniform"]
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
            elif method == "searchwithinituniform":
                if len(self.Energy.indices) == 0:
                    idx = self.random_state.choice(range(self.Energy.n))
                else:
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
                self.build_times.append(toc-tic)

            # update distances and add point to index set
            self.Energy.add(idx)
        return 
    
    def swap_phase(self, method="search", debug=False):
        assert method in ["search", "sampling"]
        k = len(self.Energy.indices)
        n = self.Energy.n

        if debug:
            inds = [self.Energy.indices[:]]
            energy_vals = [self.Energy.energy]
            dists_vals = [self.Energy.dists.copy()]
        
        if method == "search" and self.Energy.type == "lowrank":
            self.Energy.prep_search_swap() # precompute all updatings for low-rank energy
            s, w = 0, 0
            while w < n:
                if self.report_timing:
                    tic = perf_counter()
                if s in self.Energy.indices:
                    s = (s + 1) % n 
                    w += 1
                    continue 

                self.Energy.update(t=k, i=s)  # update prototype set at (k+1)th prototype spot with current s 
                U = self.Energy.prep_all_downdates(returnU=True)  # precompute all downdatings including the new prototype at index k
                
                # choose best potential prototype swap, i
                vals = U.sum(axis=1)            # prep_all_downdates() already includes the power p in the computation
                t_poss = np.where(np.isclose(vals,np.min(vals)))[0]
                t = t_poss[self.random_state.choice(len(t_poss))]
                
                assert np.isclose(vals[-1], self.Energy.energy)  # check that the last entry corresponds to current energy 
                if vals[t] < self.Energy.energy:  # did this swap improve energy?
                    self.Energy.downdate(t)

                    # interchange the t^th row and the last row to finalize this swap
                    self.Energy.indices[t] = s 
                    self.Energy.indices[-1] = -1
                    self.Energy.W[[t,-1],:] = self.Energy.W[[-1,t],:]   # downdate() already zeroes out what was the t^th row/entries
                    self.Energy.f[[t,-1]] = self.Energy.f[[-1,t]]

                    # change the Cholesky factor accordingly
                    LowRankEnergy.cholesky_add(self.Energy.L, self.Energy.G[self.Energy.indices[:k], s], t)
                    LowRankEnergy.cholesky_delete(self.Energy.L, k)
                    if self.report_timing:
                        toc = perf_counter()
                        self.swap_times.append((toc-tic, w))  # record time and number of iterations until swap
                    w = 0
                else:
                    # swap was unsuccessful, so revert the temporary update
                    self.Energy.downdate(k)
                    w += 1

                s = (s + 1) % n

            # clean up any temporary variables used in low-rank search swap
            self.Energy.end_search_swap() 

        elif method == "search": # adaptive-search swap
            t, counter = 0, 0
            while counter < n:
                if self.report_timing:
                    tic = perf_counter()
                
                # construct the eager swap vector, r_S/s_i \cup t
                r = self.Energy.compute_eager_swap_values(idx=t) 
                assert r.size == k

                # select one of the minimizers
                j_poss = np.where(r == np.min(r))[0]
                jstar = j_poss[self.random_state.choice(len(j_poss))]

                if self.report_timing:
                    toc = perf_counter()
                    self.swap_times.append((toc-tic, None)) # record time and number of iterations until swap (None for search)
                
                # if minimum swap value improves the current energy, then perform the swap and update counter
                if r[jstar] < self.Energy.energy:
                    if self.Energy.indices[jstar] != t: # make sure we are actually making a swap
                        # print("swapping", self.Energy.indices[jstar], "with", t, "old energy", self.Energy.energy, "old indices", self.Energy.indices)
                        # print(r)
                        # print(jstar, r[jstar], self.Energy.energy)
                        old_idx = self.Energy.indices[:][jstar]
                        self.Energy.swap(jstar, t)
                        # print("\tnew energy", self.Energy.energy, "new indices", self.Energy.indices, "r[jstar]", r[jstar])
                        # print("\t", r[jstar], self.Energy.energy, np.isclose(self.Energy.energy, r[jstar]))
                        if not np.isclose(self.Energy.energy, r[jstar]):
                            print(jstar, old_idx, t, "not close", len(j_poss), len(self.Energy.indices), self.Energy.indices)
                        counter = 0
                    else:
                        counter += 1
                else:
                    counter += 1
                
                # update index
                t += 1 
                t = t % n

        elif method == "sampling":   # adaptive sampling swap

            
            # initialize counters, force swap prob vector, and best prototype trackers
            t, w, u = 0, 0, 0
            p = np.zeros(k)
            best_energy, best_inds = self.Energy.energy, self.Energy.indices[:]  # will track the best found energy and indices through adaptive swap moves
            
            
            if self.report_timing:
                tic = perf_counter()
            
            # Complete at most 2*k swaps
            while u <= 2*k:
                if self.Energy.type == "lowrank":
                    U = self.Energy.prep_sampling_swap() # precompute all downdatings for low-rank energy. 
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

                else:
                    # Step 1 : update cost vector and sample a proposal prototype 
                    q_probs_wo_st = self.Energy.compute_swap_distances(idx_to_swap=t)  # includes the power p from the Energy object computation
                    q_probs_den = q_probs_wo_st.sum()
                    s_prime = self.random_state.choice(range(self.Energy.n), p=q_probs_wo_st/q_probs_den)
                    p[w] = q_probs_wo_st[self.Energy.indices[t]] /q_probs_den # probability of not swapping at this current iteration
                    w += 1
                    # Step 2: if stagnated, then terminate (p = \inf case)
                    if (s_prime == self.Energy.indices[t]) and (w == k) and (self.Energy.p is None):
                        break

                    # Step 3: If stagnated, then force a swap (p < \inf case)
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
                    if self.report_timing: # if timing, record time before it 
                        toc = perf_counter()
                        self.swap_times.append((toc-tic, w))  # record time and number of iterations until swap

                    self.Energy.swap(t, s_prime)
                    u += 1                   # increment swap counter
                    w = 0                       # reset stagnation counter
                    # check if this is best energy found so far
                    if self.Energy.energy < best_energy:
                        best_energy = self.Energy.energy
                        best_inds = self.Energy.indices[:]
                    
                    if self.report_timing:
                        tic = perf_counter()


                t = (t + 1) % k    # increment index counter


            # at the end of the swap phase, set the indices to the best found
            #     - first re-initialize the Energy object to clear out the current indices
            all_energy_values = self.Energy.energy_values[:]  # make copy of old energy values (have all the swaps recorded)
            if self.Energy.type == "cluster":
                self.Energy = ClusteringEnergy(self.Energy.X, p=self.Energy.p)
            elif self.Energy.type == "lowrank":
                self.Energy = LowRankEnergy(self.Energy.X, p=self.Energy.p)
            elif self.Energy.type == "conic":
                self.Energy = ConicHullEnergy(self.Energy.X, p=self.Energy.p)
            else:
                raise NotImplementedError(f"Energy type {self.Energy.type} not recognized...")
            
            self.Energy.init_set(best_inds)

            if debug:
                self.best_energy_vals = all_energy_values[:len(best_inds)] + self.best_energy_vals
                self.Energy.energy_values = all_energy_values + [self.Energy.energy_values[-1]]  # overwrite the energy_values list to have full history for later plotting
            if not np.isclose(self.Energy.energy, best_energy): # check that we have properly reset the Energy object
                print(best_energy, self.Energy.energy, "not same?")
            

        else:
            raise NotImplementedError(f"Method {method} not recognized for swap phase.")

        if debug:
            return inds, energy_vals, dists_vals
        return 

