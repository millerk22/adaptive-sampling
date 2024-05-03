import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from energies import *
from sampling import *
from toy import *
import pickle
import os
from argparse import ArgumentParser
import joblib
import scipy.sparse as sps

from sklearn.datasets import make_blobs

METHODS = ["passive", "p", "greedy", "p-1", "p-3", "p-5", "p-10"]
#METHODS = ["greedyla"]
def run_test(X, k, energy_type, method_strings=METHODS, seeds=np.arange(42,48),
            kernel=partial(rbf_kernel, gamma=0.1), n_jobs=12, num_la_samples=100):
    assert type(num_la_samples) in [int, float]
    if type(num_la_samples) == float:
        num_la_samples = int(X.shape[0]*num_la_samples)

    
    results = {s : defaultdict(list) for s in method_strings}
    if sps.issparse(X):
        Xfro_norm2 = sps.linalg.norm(X, ord='fro')**2.
    else:
        Xfro_norm2 = np.linalg.norm(X, ord='fro')**2.
    
    for count, method_str in enumerate(method_strings):
        print(f"Method = {method_str}, {count+1}/{len(method_strings)}")
        method_parts = method_str.split("-")
        if len(method_parts) > 1:
            method, p_val = method_parts
            p_val = float(p_val)
        else:
            method = method_str
            p_val = 2.0
        
        for seed in tqdm(seeds):
            results[method_str]["seeds"].append(seed)
            if energy_type == "cvx":
                sampler_energy = ConvexHullEnergy(X, k, n_jobs=n_jobs)
            elif energy_type == "lpkernel":
                sampler_energy = LpSubspaceEnergy(X, k, kernel=kernel)
            elif energy_type == "lp":
                sampler_energy = LpSubspaceEnergy(X, k, kernel=None)
            elif energy_type == "kmeans":
                sampler_energy = KmeansEnergy(X, k)
            else:
                print(f"Method = {method} not recognized, skipping")
                break
            if method == "passive":
                random_state = np.random.RandomState(seed)
                init_point = random_state.choice(X.shape[0])
                other_indices = np.delete(np.arange(X.shape[0]), [init_point])
                sampler_energy.init_set([init_point] + list(random_state.choice(other_indices, k-1, replace=False)))
            else:
                # perform the adaptive sampling
                adaptive_sampling(X, k, sampler_energy, method=method, seed=seed, p=p_val, num_la_samples=num_la_samples)


            results[method_str]["indices"].append(sampler_energy.indices)
            results[method_str]["energy"].append(sampler_energy.energy / Xfro_norm2)
            results[method_str]["energy_values"].append(np.array(sampler_energy.energy_values) / Xfro_norm2)
    return results



def load_dataset(dataset_name):
    if dataset_name == "blobs":
        X, _ = make_blobs(5*[500], n_features=200, cluster_std=0.1)
    elif dataset_name == "urban":
        X = np.load("./data/urban.npz")['H'].T
        print(X.shape)
        X = 1.0*X
        X /= np.max(np.max(X))
    elif dataset_name == "salinas": # HSI dataset
        X = np.load("./data/salinasa.npz")['H']
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        print(X.shape)
        labels = np.load("./data/salinasa.npz")['labels']
        labels = np.reshape(labels, (labels.shape[0]*labels.shape[1], 1)).flatten()
        print(labels.shape)
        mask = labels != 0
        X = X[mask]
        print(X.shape)
        X = 1.0 * X
        X /= np.max(np.max(X))
    elif dataset_name == "pavia":   # HSI dataset
        X = np.load("./data/pavia.npz")['H']
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        print(X.shape)
        labels = np.load("./data/pavia.npz")['labels']
        labels = np.reshape(labels, (labels.shape[0]*labels.shape[1], 1)).flatten()
        print(labels.shape)
        mask = labels != 0
        X = X[mask]
        print(X.shape)
        X = 1.0 * X
        X /= np.max(np.max(X))
        
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
    else:
        raise ValueError(f"Dataset = {dataset_name} not recognized")
    return X

POSSIBLE_ENERGIES = ['kmeans', 'lp', 'lpkernel', 'cvx', 'naive']
POSSIBLE_POLICIES = ['greedy', 'rand']
POSSIBLE_TASKS = ['kmeans', 'nmf', 'nystrom']


if __name__ == "__main__":
    parser = ArgumentParser(description="Main file for running adaptive sampling tests.")
    parser.add_argument("--dataset", default="blobs", help="Name of dataset, referenced in get_dataset function")
    parser.add_argument("--k", default=5, type=int, help="Number of samples to select")
    parser.add_argument("--resultsdir", default="./results", help="Location to save results, if args.save = 1")
    parser.add_argument("--save", default=1, type=int, help="Boolean (i.e., integer 1 or 0) of whether or not to save the results to file.")
    parser.add_argument("--energy", default="kmeans", help="String name of the evaluation energy type; one of ['kmeans', 'lp', 'lpkernel', 'cvx']")
    parser.add_argument("--numseeds", default=1, type=int, help="Number of random trials (indexed by seeds) to perform")
    parser.add_argument("--numlasamples", default=100, type=int, help="Number of samples to evaluate look ahead greedy method on")
    parser.add_argument("--postfix", default="", type=str, help="Postfix identifier string to be differentiate this run from others")
    args = parser.parse_args()
    
    assert args.energy in ['kmeans', 'lp', 'lpkernel', 'cvx']
    
    
    # load dataset and run test for the corresponding experiment name
    X = load_dataset(args.dataset)
    
    # run the test
    results = run_test(X, args.k, args.energy, seeds=np.arange(42, 42+args.numseeds), num_la_samples=args.numlasamples)
    
    
    if args.save:
        if not os.path.exists(args.resultsdir):
            os.makedirs(args.resultsdir)
        savename = os.path.join(args.resultsdir, args.dataset + "_" + args.energy + "_k" + str(args.k) + "_ns" + str(args.numseeds) + "_nla" + str(args.numlasamples) + args.postfix + ".pkl")
        print(f"Saving results to file {savename}...")
        with open(savename, 'wb') as rfile:
            pickle.dump(results, rfile)
        

