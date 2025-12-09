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
                self.times.append(toc-tic)

            # update distances and add point to index set
            self.Energy.add(idx)
        return 
    
    def swap_phase(self, method="search", max_swaps=None, debug=False):
        assert method in ["search", "sampling"]
        k = len(self.Energy.indices)
        n = self.Energy.n
        self.num_cycles = None

        if max_swaps is None:
            max_swaps = k**2

        if debug:
            inds = [self.Energy.indices[:]]
            energy_vals = [self.Energy.energy]
            dists_vals = [self.Energy.dists.copy()]

        if method == "search": # adaptive-search swap
            t, counter = 0, 0
            self.num_cycles = 0
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
                    self.times.append(toc-tic)
                
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
                if t == 0:
                    self.num_cycles += 1
                    print("cycle", self.num_cycles, "counter", counter)



        elif method == "searchold":
            count = 0
            for u in range(max_swaps):
                if self.report_timing:
                    tic = perf_counter()
                t = u % k
                
                if self.Energy.type in ["conic", "cluster"]:
                    C = np.zeros((n, k))
                    for t in range(k):
                        C[:,t] = self.Energy.compute_search_values(idx_to_swap=t) # candidates will be the unselected inds, since we consider all entries
                                                                            # of C that correspond to the current indices (S) will just have the current energy value.
                                                                            # NOTE: this gives a different value of C than in the compute_C_matrix() setting, but the minimum 
                                                                            #     across each row is the same, so the computation that comes after is unaffected.
                elif self.Energy.type == "cluster-dense":
                    C = self.Energy.compute_C_matrix()
                else:
                    raise NotImplementedError(f"search swap moves not yet implemented for {self.Energy.type}...")
                
                # find swap move in argmin_{idx,t} C(idx,t). arbitrarily break ties
                idx_poss, t_poss = np.where(C == np.min(C))

                # if the minimum is the current energy, then no swap results in a better energy, so stop swapping
                if np.intersect1d(self.Energy.indices, idx_poss).size > 0:
                    break 

                jstar = self.random_state.choice(len(idx_poss))
                idx, t = idx_poss[jstar], t_poss[jstar]

                if self.report_timing:
                    toc = perf_counter()
                    self.times.append(toc-tic)

                self.Energy.swap(t, idx)
                
                count += 1
                if count > max_swaps:
                    break
        elif method == "sampling":
            # adaptive sampling swap
            t, w = 0, 0
            p = np.zeros(k)

            # will track the best found energy and indices through adaptive swap moves
            best_energy, best_inds = self.Energy.energy, self.Energy.indices[:]
            num_forced_swaps = 0
            num_swaps_all = 0
            swap_force_probs, forced_iters = [], []
            self.best_energy_vals = [best_energy]
            for u in range(max_swaps):
                
                if self.report_timing:
                    tic = perf_counter()
                q_probs_wo_st = self.Energy.compute_swap_distances(idx_to_swap=t) # power p included in Energy object computation
                q_probs_den = q_probs_wo_st.sum()
                s_prime = self.random_state.choice(range(self.Energy.n), p=q_probs_wo_st/q_probs_den)
                p[w] = q_probs_wo_st[self.Energy.indices[t]] /q_probs_den # probability of not swapping at this current iteration
                swap_force_probs.append(p[w])
                forced_iters.append(0)
                w += 1
                swapped = False
                if s_prime != self.Energy.indices[t]: # "natural" swap
                    self.Energy.swap(t, s_prime)
                    w = 0
                    swapped = True
                elif w == k and self.Energy.p is None:  # adaptive selection swap will not occur going forward, so break
                    break
                elif w == k:
                    probs = np.concatenate(([1], np.cumprod(p)[:-1])) * (1. - p) / (1. - np.prod(p))
                    i = self.random_state.choice(range(k), p=probs)

                    t = (t + i) % k
                    force_swap_probs = q_probs_wo_st/(q_probs_den - q_probs_wo_st[self.Energy.indices[t]])
                    force_swap_probs[self.Energy.indices[t]] = 0.0
                    s_prime = self.random_state.choice(range(self.Energy.n), p=force_swap_probs)
                    self.Energy.swap(t, s_prime)
                    w = 0
                    swapped = True
                    num_forced_swaps += 1
                    forced_iters = forced_iters[:-k] + [1]*k  # overwrite the last k entries as leading to forced swaps

                if swapped:
                    # check if this is best energy found so far
                    if self.Energy.energy < best_energy:
                        best_energy = self.Energy.energy
                        best_inds = self.Energy.indices[:]
                    num_swaps_all += 1
                
                self.best_energy_vals.append(best_energy) # record best_energy_value 
                
                t = (t + 1) % k

            # at the end of the swap phase, set the indices to the best found
            #     - first re-initialize the Energy object to clear out the current indices
            all_energy_values = self.Energy.energy_values[:]  # make copy of old energy values (have all the swaps recorded)
            if self.Energy.type == "cluster-dense":
                self.Energy = ClusteringEnergyDense(self.Energy.X, p=self.Energy.p)
            elif self.Energy.type == "conic":
                self.Energy = ConicHullEnergy(self.Energy.X, p=self.Energy.p)
            elif self.Energy.type == "cluster":
                self.Energy = ClusteringEnergy(self.Energy.X, p=self.Energy.p)
            else:
                raise NotImplementedError(f"Energy type {self.Energy.type} not recognized...")
            self.Energy.init_set(best_inds)

            if debug:
                self.best_enegy_vals = all_energy_values[:len(best_inds)] + self.best_energy_vals
                self.Energy.energy_values = all_energy_values + [self.Energy.energy_values[-1]]  # overwrite the energy_values list to have full history for later plotting
            if self.Energy.energy != best_energy: # check that we have properly reset the Energy object
                print(best_energy, self.Energy.energy, "not same?")
            # print("Number of forced swaps during adaptive-sampling swap phase: ", num_forced_swaps)
            self.num_forced_swaps = num_forced_swaps  
            self.swap_force_probs = swap_force_probs
            self.forced_iters = forced_iters
            self.num_actual_swaps = num_swaps_all

            # print(len(self.forced_iters), len(forced_iters), len(self.swap_force_probs))

        else:
            # adaptive-sampling swap (old)
            check_change_flag = False
            if self.Energy.p is None:
                no_change_count = 0
                check_change_flag = True
            
            # will track the best found energy and indices through adaptive swap moves
            best_energy, best_inds = self.Energy.energy, self.Energy.indices[:]

            for u in range(max_swaps):
                if self.report_timing:
                    tic = perf_counter()
                t = u % k
                if check_change_flag:
                    old_idx = self.Energy.indices[t]

                # compute the distances to prototypes minus the current index swapped out
                q_probs_wo_st = self.Energy.compute_swap_distances(idx_to_swap=t)
                inds_wo_st = self.Energy.indices[:t] + self.Energy.indices[t+1:]
                q_probs_wo_st[inds_wo_st] = 0.0       # don't want to give any probability to those points that are already in the current indices set
                
                # sample the swap move point
                if self.Energy.p is None:
                    max_idxs = np.where(q_probs_wo_st == 1.0)[0]
                    idx = self.random_state.choice(max_idxs)
                else:
                    idx = self.random_state.choice(range(self.Energy.n), p=q_probs_wo_st/q_probs_wo_st.sum())
                
                if self.report_timing:
                    toc = perf_counter()
                    self.times.append(toc-tic)
                
                # update the Energy object according to this swap move
                self.Energy.swap(t, idx) 

                # check if this is best energy found so far
                if self.Energy.energy < best_energy:
                    best_energy = self.Energy.energy
                    best_inds = self.Energy.indices[:]
                
                # check if the index set has not changed for k consecutive iterations
                if check_change_flag:
                    if idx == old_idx:
                        no_change_count += 1 
                    else:
                        no_change_count = 0
                    old_idx = idx   
                    if no_change_count == k:
                        break
                
                if debug:
                    inds.append(self.Energy.indices[:])
                    energy_vals.append(self.Energy.energy)
                    dists_vals.append(self.Energy.dists.copy())
            

            # at the end of the swap phase, set the indices to the best found
            #     - first re-initialize the Energy object to clear out the current indices
            if self.Energy.type == "cluster-dense":
                self.Energy = ClusteringEnergyDense(self.Energy.X, p=self.Energy.p)
            elif self.Energy.type == "conic":
                self.Energy = ConicHullEnergy(self.Energy.X, p=self.Energy.p)
            elif self.Energy.type == "cluster":
                self.Energy = ClusteringEnergy(self.Energy.X, p=self.Energy.p)
            else:
                raise NotImplementedError(f"Energy type {self.Energy.type} not recognized...")
            self.Energy.init_set(best_inds)
            assert self.Energy.energy == best_energy # check that we have properly reset the Energy object
            



        if debug:
            return inds, energy_vals, dists_vals
        return 

