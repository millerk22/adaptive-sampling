import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from energies import *
from sampling import *
import pickle
import os
from argparse import ArgumentParser

from sklearn.datasets import make_blobs


# the hard-coded list of methods to run for the tests
METHODS = ["kmeans-rand", "kmeans-greedy", "lpkernel-rand", "lpkernel-greedy", "cvx-rand", "cvx-greedy"]
# hard-coded list of random seeds for the tests
SEEDS = np.arange(42,48)


def run_test(X, k, eval_energy_type, method_strings=["cvx-rand", "cvx-greedy"], seeds=np.arange(42,48),
            kernel=partial(rbf_kernel, gamma=0.1)):
    results = {s : defaultdict(list) for s in method_strings}
    Xfro_norm2 = np.linalg.norm(X, ord='fro')**2.
    
    for count, method in enumerate(method_strings):
        sampler_energy_type, policy = method.split("-")
        greedy = policy == "greedy"
        print(f"Method = {method}, {count}/{len(method_strings)}")
        for seed in tqdm(seeds):
            if sampler_energy_type == "cvx":
                sampler_energy = ConvexHullEnergy(X, k)
                adaptive_sampling(X, k, sampler_energy, greedy=greedy)
            elif sampler_energy_type == "lpkernel":
                sampler_energy = LpSubspaceEnergy(X, k, kernel=kernel)
                adaptive_sampling(X, k, sampler_energy, greedy=greedy)
            elif sampler_energy_type == "lp":
                sampler_energy = LpSubspaceEnergy(X, k, kernel=None)
                adaptive_sampling(X, k, sampler_energy, greedy=greedy)
            elif sampler_energy_type == "kmeans":
                sampler_energy = KmeansEnergy(X, k)
                adaptive_sampling(X, k, sampler_energy, greedy=greedy)
            else:
                print(f"Method = {method} not recognized, skipping")
                break
            
            results[method]["seeds"].append(seed)
            results[method]["indices"].append(sampler_energy.indices)
            
            if eval_energy_type == sampler_energy_type:
                results[method]["eval_energy"].append(sampler_energy.energy / Xfro_norm2)
                
            # in the case that the sampler is based on a different energy than the evaluation energy, we evaluate for the selected set of indices
            else: 
                # CURRRENTLY implemented to do sequential, but can CHANGE the Energy objects to evaluate for a given set.
                if eval_energy_type == "cvx":
                    eval_energy = ConvexHullEnergy(X, k)
                elif eval_energy_type == "lpkernel":
                    eval_energy = LpSubspaceEnergy(X, k, kernel=kernel)
                elif eval_energy_type == "lp":
                    eval_energy = LpSubspaceEnergy(X, k, kernel=None)
                elif eval_energy_type == "kmeans":
                    eval_energy = KmeansEnergy(X, k)
                else:
                    print("Something Wrong")
                    break
                for i in sampler_energy.indices:
                    eval_energy.add(i)

                results[method]["eval_energy"].append(eval_energy.energy / Xfro_norm2)
    return results


def load_dataset(dataset_name):
    if dataset_name == "blobs":
        X, _ = make_blobs(5*[500], n_features=200, cluster_std=0.1)
        return X
    elif dataset_name == "salinas": # HSI dataset
        X = np.load("./data/salinasa.npz")['H']
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        X = 1.0 * X
        return X 
    elif dataset_name == "pavia":   # HSI dataset
        X = np.load("./data/pavia.npz")['H']
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        X = 1.0 * X
        return X 
    else:
        raise ValueError(f"Dataset = {dataset_name} not recognized")

POSSIBLE_ENERGIES = ['kmeans', 'lp', 'lpkernel', 'cvx']
POSSIBLE_POLICIES = ['greedy', 'rand']





if __name__ == "__main__":
    parser = ArgumentParser(description="Main file for running adaptive sampling tests.")
    parser.add_argument("--dataset", default="blobs", help="Name of dataset, referenced in get_dataset function")
    parser.add_argument("--k", default=5, help="Number of samples to select")
    parser.add_argument("--resultsdir", default="./results", help="Location to save results, if args.save = 1")
    parser.add_argument("--save", default=1, type=int, help="Boolean (i.e., integer 1 or 0) of whether or not to save the results to file.")
    parser.add_argument("--evalenergy", default="kmeans", help="String name of the evaluation energy type; one of ['kmeans', 'lp', 'lpkernel', 'cvx']")
    args = parser.parse_args()
    
    assert args.evalenergy in ['kmeans', 'lp', 'lpkernel', 'cvx']
    assert np.array([(es in POSSIBLE_ENERGIES) & (p in POSSIBLE_POLICIES) for es, p in METHODS.split('-')]).all()
    
    
    # load dataset and run test for the corresponding experiment name
    X = load_dataset(args.dataset)
    
    # run the test
    results = run_test(X, args.k, args.evalenergy, method_strings=METHODS, seeds=SEEDS)
    
    
    if args.save:
        if not os.path.exists(args):
            os.makedirs(args.resultsdir)
        savename = os.path.join(args.resultsdir, args.dataset + ".pkl")
        print(f"Saving results to file {savename}...")
        with open(savename, 'wb') as rfile:
            pickle.dump(results, rfile)
        

