import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from energies import ConicHullEnergy
from sampling import *
from toy import *
import pickle
import os
import yaml
from argparse import ArgumentParser
import joblib
import scipy.sparse as sps

from sklearn.datasets import make_blobs


ALL_METHODS = ["search", "sampling", "uniform"] #, "search_search", "sampling_search", "sampling_sampling", "search_sampling"]  
METHODS = ALL_METHODS #[:-1]
OVERSAMPLE_METHODS = ["sampling", "uniform"] 


def run_test(args, X, k, energy_type, p, seeds=np.arange(42,48), n_jobs=12, report_timing=False, overwrite=[],
            oversample=[], k_os=50):

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
                if len(results[method_str]['energy_values']) == args.numseeds:
                    already_done.append(method_str)
                else:
                    methods_to_do.append(method_str)
            else:
                methods_to_do.append(method_str)

        for method_str in methods_to_do:
            results[method_str] = defaultdict(list)
    

    # order these by alphabetical order
    methods_to_do = sorted(methods_to_do)
    

    print("Already found results for: ", ", ".join(already_done))
    print("(Re-)Computing results for: ", ", ".join(methods_to_do))
    print(f"\toversample methods = {OVERSAMPLE_METHODS}")
    print("="*40)
    
    for count, method_str in enumerate(methods_to_do):
        if len(results[method_str]['energy']) == args.numseeds:
            print(f"Already found {method_str} result in {savename}, skipping...")
            continue

        print(f"Overall Method = {method_str}, {count+1}/{len(methods_to_do)}")
        if len(method_str.split("_")) == 2:
            build_method, swap_method = method_str.split("_")
        else:
            build_method = method_str
            swap_method = None
        
        if build_method in OVERSAMPLE_METHODS: # if this build method is designated to be oversampled, set the value of k accordingly
            k_todo = k_os 
        else:
            k_todo = k

        for i, seed in tqdm(enumerate(seeds), total=len(seeds)):
            results[method_str]["seeds"].append(seed)

            if swap_method is None:
                # Instantiate an Energy object for this test
                if energy_type == "conic":
                    Energy = ConicHullEnergy(X, k_todo, p=p, n_jobs=n_jobs)
                else:
                    print(f"Energy type = {energy_type} not recognized, skipping")
                    break

                # Perform the build function for this run
                if build_method == "uniform":
                    # uniform can be done all at once
                    random_state = np.random.RandomState(seed)
                    Energy.init_set(list(random_state.choice(Energy.n, k_todo, replace=False)))
                    if report_timing:
                        results[method_str]["times"].append(k_todo*[None])
                else:
                    if build_method == "sampling":
                        sampler = AdaptiveSampling(Energy, seed=seed, report_timing=report_timing)
                    elif build_method == "search":
                        sampler = AdaptiveSearch(Energy, seed=seed, report_timing=report_timing)
                    else:
                        raise NotImplementedError(f"build_method = {build_method} not recognized...")

                    sampler.build_phase(k_todo)
                    if report_timing:
                        results[method_str]["times"].append(sampler.times)

                results[method_str]["indices"].append(Energy.indices)
                results[method_str]["energy"].append(Energy.energy)
                
            else:
                # if we have a swap_method string, then we will read in previously done build moves to 
                # avoid recomputing...
                raise NotImplementedError("Swap moves not implemented yet...")
                indices_from_as = results[build_method_str]["indices"][i] # get the previously computed indices from the non-swap moves method
                                                                    # should have because of check done in main part of script
                energy_values, indices, times_all = perform_swaps_for_all(X, args.kstart, indices_from_as, swap_method, report_timing=True, verbose=True)
                results[method_str]["indices"].append(indices)
                results[method_str]["energy_values"].append(np.array(energy_values) / Xfro_norm2)
                if report_timing:
                    # add the time from as_method to select each of the k points via adaptive sampling and then the time to do all the swap moves thereafter
                    results[method_str]["times"].append([sum(results[as_method_str]["times"][i][:idx]) + sum(times_all[idx]) for idx in range(args.kstart, len(indices_from_as))])


        if args.save:
            print(f"Saving (intermediate) results to file {savename}...")
            with open(savename, 'wb') as rfile:
                pickle.dump(results, rfile)
    return 



def load_dataset(dataset_name, n_test=500):
    if dataset_name == "blobssmallest":
        X, _ = make_blobs(5*[50], n_features=200, cluster_std=0.1)
    elif dataset_name == "blobssmall":
        X, _ = make_blobs(5*[100], n_features=200, cluster_std=0.1)
    elif dataset_name == "blobs":
        X, _ = make_blobs(5*[500], n_features=200, cluster_std=0.1)
    elif dataset_name == "urban":
        X = np.load("./data/urban.npz")['H'].T
        X = 1.0*X
        X -= min(0.0, X.min())
        X /= np.max(np.max(X))
    elif dataset_name == "urbansub":
        X = np.load("./data/urban.npz")['H'].T
        X = 1.0*X
        X -= min(0.0, X.min())
        X /= np.max(np.max(X))
        rstate = np.random.RandomState(42)
        subset = rstate.choice(X.shape[0], 7500, replace=False)
        print(X.shape)
        X = X[subset]
        print(X.shape)
    elif dataset_name == "salinas": # HSI dataset
        X = np.load("./data/salinasa.npz")['H']
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        labels = np.load("./data/salinasa.npz")['labels']
        labels = np.reshape(labels, (labels.shape[0]*labels.shape[1], 1)).flatten()
        mask = labels != 0
        X = X[mask]
        X = 1.0 * X
        X -= min(0.0, X.min())
        X /= np.max(np.max(X))
        print(X.min(), X.max())
        print(X.shape)
    elif dataset_name == "pavia":   # HSI dataset
        X = np.load("./data/pavia.npz")['H']
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        labels = np.load("./data/pavia.npz")['labels']
        labels = np.reshape(labels, (labels.shape[0]*labels.shape[1], 1)).flatten()
        mask = labels != 0
        X = X[mask]
        X = 1.0 * X
        X -= min(0.0, X.min())
        X /= np.max(np.max(X))
        print(X.shape)
    
    elif dataset_name == "paviasub":   # HSI dataset
        X = np.load("./data/pavia.npz")['H']
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        labels = np.load("./data/pavia.npz")['labels']
        labels = np.reshape(labels, (labels.shape[0]*labels.shape[1], 1)).flatten()
        mask = labels != 0
        X = X[mask]
        X = 1.0 * X
        X -= min(0.0, X.min())
        X /= np.max(np.max(X))
        rstate = np.random.RandomState(42)
        subset = rstate.choice(X.shape[0], 5000, replace=False)
        print(X.shape)
        X = X[subset]
        print(X.shape)
        
    elif dataset_name == "articles":
        (X, _, _) = joblib.load("../topic-model-tutorial/articles-tfidf.pkl")
    elif dataset_name == "snp":
        labels = np.load("snps/labels.npy", allow_pickle=True)
        X = np.load("snps/data.npy", allow_pickle=True)
        ordering = np.argsort(labels)
        X, labels = X[ordering], labels[ordering]
        nan_inds = np.where(np.isnan(X))
        col_means = np.nanmean(X, axis=0)
        X[nan_inds] = np.take(col_means, nan_inds[1]) 
    elif dataset_name == "smile":
        X, bw = smile(10000)
    elif dataset_name == "outliers":
        X, _ = outliers(10000)
    elif dataset_name == "test":
        rand_state_ = np.random.RandomState(42)
        n, d, ktrue = n_test, 100, 5
        noise = 0.04
        hthresh = 0.3
        Wtrue = rand_state_.rand(d, ktrue) 
        Htrue = rand_state_.rand(ktrue, n)
        Htrue /= Htrue.sum(axis=0).reshape(1, n)
        Htrue *= np.maximum(rand_state_.rand(1, n), hthresh)
        N = rand_state_.randn(d, n)*noise
        X = Wtrue @ Htrue + N
        X = X.T
        X[X <= 0.0] = 0.0
    else:
        raise ValueError(f"Dataset = {dataset_name} not recognized")
    return X.T



if __name__ == "__main__":
    parser = ArgumentParser(description="Main file for running adaptive sampling tests.")
    parser.add_argument("--dataset", default="blobs", help="Name of dataset, referenced in get_dataset function")
    parser.add_argument("--k", default=10, type=int, help="Number of samples to select")
    parser.add_argument("--k_oversample", default=50, type=int, help="Number of samples to select via oversampling")
    parser.add_argument("--resultsdir", default="./results", help="Location to save results, if args.save = 1")
    parser.add_argument("--save", default=1, type=int, help="Boolean (i.e., integer 1 or 0) of whether or not to save the results to file.")
    parser.add_argument("--energy", default="conic", help="String name of the evaluation energy type; one of ['conic']")
    parser.add_argument("--numseeds", default=1, type=int, help="Number of random trials (indexed by seeds) to perform")
    parser.add_argument("--postfix", default="", type=str, help="Postfix identifier string to be differentiate this run from others")
    parser.add_argument("--time", default=1, type=int, help="Bool flag (0 or 1) of whether or not to record times for each iteration of methods.")
    parser.add_argument("--njobs", default=12, type=int, help="Number of CPU cores to utilize in parallelization.")
    parser.add_argument("--config", default="", type=str, help="Location of .yml configuration file containing 'methods' list.")
    parser.add_argument("--kstart", default=2, type=int, help="Start value of k to perform swap moves.")
    parser.add_argument("--ntest", default=500, type=int, help="Size of 'test' dataset for timing comparisons.")
    args = parser.parse_args()

    
    overwrite_methods = [] 
    args.powers = [1, 2, None]

    if args.config != "":
        assert os.path.exists(args.config)
        with open(args.config,"r") as file:
            config = yaml.safe_load(file)
        if 'overwrite' in config:
            overwrite_methods = sorted(config['overwrite'])
        else:
            overwrite_methods = []
        assert np.isin(overwrite_methods, ALL_METHODS).all()

        if 'k' in config:
            args.k = config['k']
        if 'k_oversample' in config:
            args.k_oversample = config['k_oversample']
        if 'dataset' in config:
            args.dataset = config['dataset']
        if 'energy' in config:
            args.energy = config['energy']
        if 'time' in config:
            args.time = int(config['time'])
        if 'numseeds' in config:
            args.numseeds = config['numseeds']
        if 'powers' in config:
            args.powers = config['powers']

    assert args.energy in ['conic']

    # load dataset and run test for the corresponding experiment name
    X = load_dataset(args.dataset, n_test=args.ntest)

    if args.dataset == "test":
        args.dataset = "test" + str(args.ntest)
    
    # run the test
    print(f"------------ Running Test for {args.dataset} ----------------")
    print(f"\tn = {X.shape[0]}, k = {args.k}, k_oversample = {args.k_oversample}, numseeds = {args.numseeds}")
    
    for p in args.powers:
        print()
        print("=================================================")
        print(f"================== p = {p} =======================")
        print("=================================================")

        run_test(args, X, args.k, args.energy, p, seeds=np.arange(42,42+args.numseeds), \
                 report_timing=args.time, n_jobs=args.njobs, overwrite=overwrite_methods, k_os=args.k_oversample)