import numpy as np
from sklearn.utils.extmath import stable_cumsum
from functools import partial
from energies import *
from time import perf_counter


def adaptive_sampling(X, k, Energy, p_init=None, seed=42, method='greedy', p=2.0, swap_method=None, 
                      num_la_samples=-1, max_swaps=None, report_timing=False):
    """
    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n, d)
        The data to pick seeds for.

    k : int
        The number of seeds to choose.
        
    Energy : Energy object
        Python object that tracks the values of the individual "distances" and the overall energy 
        resulting from the selected inputs.

    seed : int
        Random state seed for reproducibility

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    centers : ndarray of shape (k, d)
        The initial centers for k-means.

    indices : ndarray of shape (k,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    assert method in ['greedy', 'p', 'greedyla']
    assert abs(num_la_samples) > 0 
    n, d = X.shape
    assert num_la_samples < n 
    random_state = np.random.RandomState(seed)

    samples = np.empty((k, d), dtype=X.dtype)

    # Pick first center randomly and track index of point
    sample_id = random_state.choice(n, p=p_init)
    # Initialize squared distances
    Energy.add(sample_id)

    # Pick the remaining k-1 points
    times = []
    for c in range(1, k):
        # make sure Energy.dists and Energy.energy are non-negative
        Energy.dists[Energy.dists <= 0] = 0.0
        Energy.energy = Energy.dists.sum()
        
        if report_timing:
            t0 = perf_counter()
        if method == 'greedy':
            sample_id = np.argmax(Energy.dists)
            Energy.add(sample_id)
        elif method == 'p':
            if p == 2.0:
                rand_vals = random_state.uniform(size=1) 
                dists_p = Energy.dists / Energy.energy
            else:
                dists_p = (Energy.dists / Energy.energy)**(p/2.0)
                rand_vals = random_state.uniform(size=1) * (dists_p.sum())

            candidate_ids = np.searchsorted(
                stable_cumsum(dists_p), rand_vals
            )
            # numerical imprecision can result in a candidate_id out of range
            np.clip(candidate_ids, None, Energy.n - 1, out=candidate_ids)
            
            # choose maximizer of the candidates
            sample_id = candidate_ids[np.argmax(Energy.dists[candidate_ids])]
            
            # update squared distances
            Energy.add(sample_id)
            
        elif method == 'greedyla':
            if num_la_samples > 0:
                candidate_inds = random_state.choice(np.delete(np.arange(Energy.n), Energy.indices), num_la_samples, replace=False)
            else:
                candidate_inds = np.delete(np.arange(Energy.n), Energy.indices)
            
            candidate_energy_vals = Energy.look_ahead(candidate_inds)
            sample_id  = candidate_inds[np.argmin(candidate_energy_vals)]
            Energy.add(sample_id)
            
        else:
            raise ValueError(f"Method {method} not recognized...")
        
        if report_timing:
            t1 = perf_counter()
            times.append(t1-t0)
    
    

    if swap_method:
        assert swap_method.split("-")[0] in ["greedyla", "p", "greedy"]
        if max_swaps is None:
            max_swaps = k**2 //2
        assert max_swaps == int(max_swaps)


        if report_timing:
            times.append(-1) # put in a negative value to delineate the times for adaptive sampling steps and the swap moves afterward
            
        
        no_change_count = 0
        for u in range(max_swaps):
            if report_timing:
                t0 = perf_counter()
            swap_flag = Energy.swap_move(swap_method, j_adap=u % k, verbose=True)
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

    if report_timing:
        return times
    
    return 

