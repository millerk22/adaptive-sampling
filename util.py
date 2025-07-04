import numpy as np
import os 
import pickle
from tqdm import tqdm
from collections import defaultdict

from energies import *
from sampling import *

from sklearn.metrics import pairwise_distances
from scipy.optimize import minimize


ALL_METHODS = ["search", "sampling", "uniform", "search_search", "sampling_sampling", "sampling_search"]  # 
METHODS = ALL_METHODS   # search swap moves is taking a very long time doing all i <= k swaps....
OVERSAMPLE_METHODS = ["sampling", "uniform"] 


def find_methods_to_do(args, p, overwrite=[]):
    if args.save:
        if not os.path.exists(args.resultsdir):
            os.makedirs(args.resultsdir)
        if p is None:
            pstring = 'inf'
        else:
            pstring = str(p)
        savename = os.path.join(args.resultsdir, args.dataset + "_" + args.energy + "_k" + str(args.k) + "_p" + str(pstring) + "_ns" + str(args.numseeds) + args.postfix + ".pkl")
        results = None 
        if os.path.exists(savename):
            with open(savename, "rb") as f:
                results = pickle.load(f)

    if results is None:
        results = {s : defaultdict(list) for s in METHODS}
        methods_to_do = METHODS 
        already_done = []
    else:
        methods_to_do = []
        already_done = []
        for method_str in METHODS:
            if method_str in overwrite:
                methods_to_do.append(method_str)
                continue 
            if method_str in results:
                if len(results[method_str]['seeds']) == args.numseeds:
                    already_done.append(method_str)
                elif (method_str.split("_")[0] == "search") and (len(results[method_str]['seeds']) == 1):
                    already_done.append(method_str)
                elif (method_str == "sampling_search") and (len(results[method_str]['seeds']) == 1):
                    already_done.append(method_str)
                else:
                    methods_to_do.append(method_str)
            else:
                methods_to_do.append(method_str)

        for method_str in methods_to_do:
            results[method_str] = defaultdict(list)
    

    # order these by alphabetical order
    methods_to_do = sorted(methods_to_do)

    return already_done, methods_to_do, savename, results


"""
Need to have the experiments run build phase and swap phase separately, since for swap methods we will be 
    figuring out the swaps for each 1<= i <= k, not just always doing the k build and then swaps.
"""

def run_experiment(X, p, labels, method_str, results, seeds, args):
    if len(method_str.split("_")) == 2:
        build_method, swap_method = method_str.split("_")
    else:
        build_method = method_str
        swap_method = None
    
    # if this build method is designated to be oversampled, set the value of k accordingly
    if build_method in OVERSAMPLE_METHODS: 
        k_todo = args.k_oversample 
    else:
        k_todo = args.k

    for i, seed in tqdm(enumerate(seeds), total=len(seeds)):
        # since search is deterministic, only need to run one test
        if i > 0:
            if method_str.split("_")[-1] == "search":  # this does include sampling_search, but for computational considerations we'll just do on the first seed.
                continue 
        
        results[method_str]["seeds"].append(seed)

        if swap_method is None:
            # Instantiate an Energy object for this test
            if args.energy == "conic":
                Energy = ConicHullEnergy(X, p=p, n_jobs=args.njobs, verbose=True)
            elif args.energy == "cluster":
                Energy = ClusteringEnergy(X, p=p)
            elif args.energy == "cluster-dense":
                Energy = ClusteringEnergyDense(X, p=p)
            else:
                print(f"Energy type = {args.energy} not recognized, skipping")
                break
            
            # Perform the build function for this run
            if build_method == "uniform":
                # uniform can be done all at once
                random_state = np.random.RandomState(seed)
                Energy.init_set(list(random_state.choice(Energy.n, k_todo, replace=False)))
                if args.time:
                    results[method_str]["times"].append(k_todo*[None])
            else:
                sampler = AdaptiveSampler(Energy, seed=seed, report_timing=args.time)
                sampler.build_phase(k_todo, method=build_method)
                if args.time:
                    results[method_str]["times"].append(sampler.times)

            results[method_str]["indices"].append(Energy.indices)
            results[method_str]["energy"].append(Energy.energy)
            results[method_str]["energy_values"].append(Energy.energy_values)
            
        else:
            # if we have a swap_method string, then we will read in previously done build moves to 
            # avoid recomputing...
            all_build_inds = results[build_method]["indices"][i] # get the previously computed indices from the non-swap moves method
                                                                # should have because of check done in main part of script

            indices_swap = []
            energy_swap = []
            energy_values_swap = []
            times_swap = []
            for k_ in range(1, args.k+1):
                # Instantiate an Energy object for this test
                if args.energy == "conic":
                    Energy = ConicHullEnergy(X, p=p, n_jobs=args.njobs, verbose=True)
                elif args.energy == "cluster":
                    Energy = ClusteringEnergy(X, p=p)
                elif args.energy == "cluster-dense":
                    Energy = ClusteringEnergyDense(X, p=p)
                else:
                    print(f"Energy type = {args.energy} not recognized, skipping")
                    break

                # initialize with k_ points, will do swaps from here
                Energy.init_set(all_build_inds[:k_])

                # instantiate adaptive sampler
                sampler = AdaptiveSampler(Energy, seed=seed, report_timing=args.time)
                sampler.swap_phase(method=swap_method, max_swaps=k_**2)

                indices_swap.append(Energy.indices)
                energy_swap.append(Energy.energy)
                energy_values_swap.append(Energy.energy_values)
                if args.time:
                    times_swap.append(sampler.times)

            results[method_str]["indices"].append(indices_swap)
            results[method_str]["energy"].append(energy_swap)
            results[method_str]["energy_values"].append(energy_values_swap)
            if args.time:
                # add the time from as_method to select each of the k points via adaptive sampling and then the time to do all the swap moves thereafter
                results[method_str]["times"].append(times_swap)

    return results 





# Code to perform analogue of Lloyd's algorithm for p >= 1 (not just p = 2) in Clustering cas

def euclidean_p_center(points, p, tol=1e-4):
    def objective(c):
        diff = points - c
        dists = np.linalg.norm(diff, axis=1)
        if p is not None:
            return np.sum(dists**p)
        return dists[np.argmax(dists)]
    
    c0 = np.mean(points, axis=0)  # initial guess
    result = minimize(objective, c0, method='L-BFGS-B', tol=tol)
    return result.x

def euclidean_p_kmeans(X, centers=None, p=2, k=10, max_iter=100, tol=1e-4):
    n, d = X.shape
    if centers is None:
        centers = X[np.random.choice(n, k, replace=False)]
    else:
        k = centers.shape[0]

    for _ in range(max_iter):
        # Assignment step: distances^p using Euclidean norm (M step)
        D = pairwise_distances(X, centers, metric='euclidean')
        labels = np.argmin(D, axis=1)

        # Update centers step (E step)
        new_centers = []
        for l in range(k):
            cluster_points = X[labels == l]
            if len(cluster_points) == 0:
                new_centers.append(X[np.random.choice(n)])
            else:
                c_l = euclidean_p_center(cluster_points, p)
                new_centers.append(c_l)
        new_centers = np.vstack(new_centers)

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break

    return centers, labels


def get_reference_inds(X, labels, p=2):
    k = np.unique(labels).size 

    # get initial centroids from ground truth labelings 
    mus = []
    for i in np.unique(labels):
        i_inds = np.where(labels == i)[0].tolist()
        mu = euclidean_p_center(X[i_inds], p=p)
        mus.append(mu)
    
    # iterate with Lloyd's algorithm to further refine centroids
    centers_p, labels_p = euclidean_p_kmeans(X, centers=np.array(mus), p=p)
    reference_inds = []
    for i in np.unique(labels_p):
        i_inds = np.where(labels_p == i)[0]

        # choose the point that is closest each centroid to represent it in our reference_indices
        reference_inds.append(i_inds[np.argmin(pairwise_distances(X[i_inds], centers_p[i,:].reshape(1,-1)))]) 
    return reference_inds
