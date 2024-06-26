import numpy as np
from sklearn.utils.extmath import stable_cumsum
from functools import partial
from energies import *


def adaptive_sampling(X, k, Energy, p_init=None, seed=42, method='greedy', p=2.0, num_la_samples=100):
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
    n, d = X.shape
    
    random_state = np.random.RandomState(seed)

    samples = np.empty((k, d), dtype=X.dtype)

    # Pick first center randomly and track index of point
    sample_id = random_state.choice(n, p=p_init)
    
    # Initialize squared distances
    Energy.add(sample_id)

    # Pick the remaining k-1 points
    for c in range(1, k):
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
            candidate_inds = random_state.choice(np.delete(np.arange(Energy.n), Energy.indices), num_la_samples, replace=False)
            results_dict = Energy.look_ahead(candidate_inds)
            energy_vals = np.array([results_dict[c]['energy'] for c in results_dict])
            sample_id  = candidate_inds[np.argmin(energy_vals)]
            Energy.update_from_look_ahead(sample_id, results_dict[sample_id])
            del results_dict
            
        else:
            raise ValueError(f"Method {method} not recognized...")
        
        

    return 

