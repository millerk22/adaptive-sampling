import numpy as np
from sklearn.utils.extmath import stable_cumsum
from functools import partial
from energies import *

   
def adaptive_sampling(X, k, Energy, p_init=None, seed=42, n_local_trials=1, greedy=False):
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

    indices : ndarray of sh|ape (k,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n, d = X.shape
    
    random_state = np.random.RandomState(seed)

    samples = np.empty((k, d), dtype=X.dtype)

    # Set the number of local seeding trials if none is given. (Copied from sklearn kmeans_plus_plus code)
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(k))

    # Pick first center randomly and track index of point
    sample_id = random_state.choice(n, p=p_init)
    
    # Initialize squared distances
    Energy.add(sample_id)

    # Pick the remaining k-1 points
    for c in range(1, k):
        if not greedy:
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            rand_vals = random_state.uniform(size=n_local_trials) * Energy.energy
            candidate_ids = np.searchsorted(
                stable_cumsum(Energy.dists), rand_vals
            )
            # numerical imprecision can result in a candidate_id out of range
            np.clip(candidate_ids, None, Energy.n - 1, out=candidate_ids)

            # choose maximizer of the candidates
            sample_id = candidate_ids[np.argmax(Energy.dists[candidate_ids])]
        else:
            sample_id = np.argmax(Energy.dists)
        
        # update squared distances
        Energy.add(sample_id)

    return 
