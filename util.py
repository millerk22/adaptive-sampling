import numpy as np
import os 
import pickle
from tqdm import tqdm
from collections import defaultdict

from energies import *
from sampling import *

from sklearn.metrics import pairwise_distances
from scipy.optimize import minimize


def find_methods_to_do(args, p):
    savename = None
    if args.save:
        # if results directory doesn't exist, create it
        if not os.path.exists(args.resultsdir):
            os.makedirs(args.resultsdir)
        
        # string for p value in the of p = \infty (None)
        if p is None:
            pstring = 'inf'
        else:
            pstring = str(p)
        
        # define the save name for the results, name includes some experiment parameters for identifying
        savename = os.path.join(args.resultsdir, args.dataset + "_" + args.energy + "_k" + str(args.k) + "_p" + str(pstring) + "_ns" + str(args.numseeds) + args.postfix + ".pkl")
        results = None 
        if os.path.exists(savename):
            with open(savename, "rb") as f:
                results = pickle.load(f)
    

    # find what methods we have not run yet and need to do
    if results is None:
        results = {s : defaultdict(list) for s in args.methods}
        methods_to_do = args.methods 
        already_done = []
    else:
        methods_to_do = []
        already_done = []
        for method_str in args.methods:
            if method_str in args.overwrite:
                methods_to_do.append(method_str)
                continue 
            if method_str in results:
                if len(results[method_str]['seeds']) == args.numseeds:
                    already_done.append(method_str)
                elif (method_str.split("_")[0] == "search") and (len(results[method_str]['seeds']) == 1):
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




def run_experiments_with_p(args, X, p, labels=None):

    already_done, methods_to_do, savename, results = find_methods_to_do(args, p)    
    
    print("Already found results for: ", ", ".join(already_done))
    print("(Re-)Computing results for: ", ", ".join(methods_to_do))
    print("\tOversample methods: ", ", ".join(args.oversample))
    print("\tOverwrite methods: ", ", ".join(args.overwrite))

    
    for count, method_str in enumerate(methods_to_do):
        if len(results[method_str]['build_values']) == args.numseeds:
            print(f"Already found {method_str} result in {savename}, skipping...")
            continue

        print(f"Overall Method = {method_str}, {count+1}/{len(methods_to_do)}")
        
        results = run_experiment(X, p, method_str, results, args, labels=labels)

        if args.save:
            print(f"Saving (intermediate) results to file {savename}...")
            with open(savename, 'wb') as rfile:
                pickle.dump(results, rfile)
    return 


"""
Note: Need to have the experiments run build phase and swap phase separately, since for swap methods we will be 
    figuring out the swaps for each 1<= i <= k, not just always doing the k build and then swaps.
"""

def run_experiment(X, p, method_str, results, args, labels=None):
    if len(method_str.split("_")) == 2:
        build_method, swap_method = method_str.split("_")
    else:
        build_method = method_str
        swap_method = None
    
    # if this build method is designated to be oversampled, set the value of k accordingly
    if build_method in args.oversample: 
        k_todo = args.k_oversample 
    else:
        k_todo = args.k

    for i, seed in tqdm(enumerate(args.seeds), total=len(args.seeds)):
        # since search is deterministic, only need to run one test
        if i > 0:
            if method_str.split("_")[-1] == "search":  # this does include sampling_search, but for computational considerations we'll just do on the first seed.
                continue 

        # a build only method
        if swap_method is None:
            # Instantiate an Energy object for this test
            if args.energy == "conic":
                Energy = ConicHullEnergy(X, p=p, n_jobs=args.njobs, verbose=True)
            elif args.energy == "cluster":
                Energy = ClusteringEnergy(X, p=p)
            elif args.energy == "cluster-nongram":
                Energy = ClusteringEnergyNonGram(X, p=p)
            elif args.energy == "lowrank":
                Energy = LowRankEnergy(X, p=p)
            else:
                print(f"Energy type = {args.energy} not recognized, skipping")
                break
            
            # Perform the build function for this run
            sampler = AdaptiveSampler(Energy, seed=seed, record=True)
            sampler.build_phase(k_todo, method=build_method)
            results[method_str]["build_times"].append(sampler.build_times)
            results[method_str]["build_values"].append(sampler.build_values)
            results[method_str]["indices"].append(Energy.indices)
            
        else:
            # if we have a swap_method string, then we will read in previously done build moves to 
            # avoid recomputing...
            indices_build = results[build_method]["indices"][i] # get the previously computed indices from the non-swap moves method
                                                                # should have because of check done in main part of script

            final_swap_values = []
            final_swap_indices = {}
            all_swap_values = {}
            all_swap_times = {}
            all_swap_stag = {}
            
            for k_ in tqdm(range(1, args.k+1), total=args.k, desc=f"Performing swaps for each of 1 to {args.k} points..."):
                # Instantiate an Energy object for this test
                if args.energy == "conic":
                    Energy = ConicHullEnergy(X, p=p, n_jobs=args.njobs, verbose=True)
                elif args.energy == "cluster":
                    Energy = ClusteringEnergy(X, p=p)
                elif args.energy == "cluster-nongram":
                    Energy = ClusteringEnergyNonGram(X, p=p)
                elif args.energy == "lowrank":
                    Energy = LowRankEnergy(X, p=p)
                else:
                    print(f"Energy type = {args.energy} not recognized, skipping")
                    break

                # initialize with k_ points, will do swaps from here
                Energy.init_set(indices_build[:k_])

                # instantiate adaptive sampler
                sampler = AdaptiveSampler(Energy, seed=seed, record=True)
                if k_ == 1 and swap_method == "search":
                    # don't do a swap since search build is already "optimal subset of size 1" 
                    print(f"Skipping swap phase with k_ = {k_} for {swap_method}...")
                    # in this case the sampler values will just be empty lists below
                elif k_ == 1 and args.energy == "cluster":
                    print(f"Skipping k_ = 1 for cluster energy")
                else:
                    sampler.swap_phase(method=swap_method, debug=False)
                final_swap_values.append(Energy.energy)
                final_swap_indices[k_] = Energy.indices
                all_swap_values[k_] = sampler.swap_values 
                all_swap_times[k_] = sampler.swap_times 
                all_swap_stag[k_] = sampler.swap_stag
                

            # record all the relevant info from each of the swap runs
            results[method_str]["swap_values"].append(final_swap_values)
            results[method_str]["indices"].append(final_swap_indices)
            results[method_str]["all_swap_values"].append(all_swap_values)
            results[method_str]["all_swap_times"].append(all_swap_times)
            results[method_str]["all_swap_stag"].append(all_swap_stag)

        # if completed this seed's experiment, append the seed value
        results[method_str]["seeds"].append(seed)

    return results 





# Code to perform analogue of Lloyd's algorithm for p >= 1 (not just p = 2) in Clustering case

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





from scipy.linalg import cholesky


def get_components_naive(G, indices):
    L = cholesky(G[np.ix_(indices, indices)], lower=True)
    W = np.linalg.inv(G[np.ix_(indices, indices)]) @ G[indices,:]
    f = np.diag(np.linalg.inv(G[np.ix_(indices, indices)])).real
    d = np.diag(G).real - np.sum(W.conj() * G[indices,:], axis=0).real
    return L, W, f, d


def check_components(energy, t=None, verbose=True, nround=4, checkR=False, extradim=False):
    if t is not None:
        indices = energy.indices[:t] + energy.indices[t+1:]
        indices_t = list(range(t)) + list(range(t+1, len(energy.indices)))
    else:
        indices = energy.indices[:]
        indices_t = list(range(len(energy.indices)))
    if extradim:
        indices = indices[:-1]
        indices_t = indices_t[:-1]
    print("inds = ", indices)
    print("G[inds,inds] symmetric = ", np.allclose(energy.G[np.ix_(indices, indices)], energy.G[np.ix_(indices, indices)].conj().T))
    print("G[inds, inds] invertible = ", np.linalg.cond(energy.G[np.ix_(indices, indices)]) < 1/np.finfo(energy.G.dtype).eps)
    L_naive, W_naive, f_naive, d_naive = get_components_naive(energy.G, indices)
    matchL = np.allclose(energy.L[np.ix_(indices_t, indices_t)], L_naive)
    matchW = np.allclose(energy.W[indices_t, :], W_naive)
    matchf = np.allclose(energy.f[indices_t], f_naive)
    matchd = np.allclose(energy.d, d_naive)
    print("L,W,f,d match: ", matchL and matchW and matchf and matchd)
    
    print("L close:", matchL)
    if not matchL and verbose:
        print("Computed L:\n", np.round(energy.L[np.ix_(indices_t, indices_t)], nround))
        print("Naive L:\n", np.round(L_naive, nround))
    print("W close:", matchW)
    if not matchW and verbose:
        print("Computed W:\n", np.round(energy.W[indices_t, :], nround))
        print("Naive W:\n", np.round(W_naive, nround))
        print("full self.W:\n", np.round(energy.W, nround))
    print("f close:", matchf)
    if not matchf and verbose:
        print("Computed f:\n", np.round(energy.f[indices_t],nround))
        print("Naive f:\n", np.round(f_naive, nround))
        print("full self.f:\n", np.round(energy.f, nround))
    print("d close:", matchd)
    if not matchd and verbose:
        print("Computed d:\n", np.round(energy.d, nround))
        print("Naive d:\n", np.round(d_naive, nround))
    
    if checkR:
        R_naive = energy.G - energy.G[:, indices] @ np.linalg.inv(energy.G[np.ix_(indices, indices)]) @ energy.G[indices, :]
        matchR = np.allclose(R_naive, energy.R)
        print("R close:", matchR)
        if not matchR and verbose:
            print("Computed R:\n", np.round(energy.R, nround))
            print("Naive R:\n", np.round(R_naive, nround))
        if hasattr(energy, 'Y'):
            Y_naive = energy.W @ energy.R
            matchY = np.allclose(Y_naive, energy.Y)
            print("Y close: ", matchY)
            if not matchY and verbose:
                print("Is close = ", np.where(np.isclose(energy.Y, Y_naive)))
                print("Computed Y:\n", np.round(energy.Y, nround))
                print("Naive Y:\n", np.round(Y_naive, nround))

    if matchL and matchW and matchf and matchd and (t is not None) and verbose:
        print("All components match naive computation!")
        print("L @ t:\n", np.round(energy.L[t, :], nround)) 
        print("W @ t:\n", np.round(energy.W[t, :], nround))
        print("f @ t:\n", np.round(energy.f[t], nround))
        print("d @ t:\n", np.round(energy.d[t], nround))
        if checkR and matchR:
            print("R @ t:\n", np.round(energy.R[:, t], nround))
    return 
