import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product
import pickle
import os
from argparse import ArgumentParser
import joblib
from nnls_FPGM import *


def run_test(X, W, method, maxiter=500, verbose=True):
    if method == "fpgm":
        return nnls_FPGM(X, W, delta=1e-3, inneriter=maxiter, alpha0=0.05, returnH=False, verbose=True)
    elif method == "ogm":
        return nnls_OGM(X, W, delta=1e-3, maxiter=maxiter, lam=1.0,returnH=False, verbose=True)
    else:
        raise NotImplementedError(f"method = '{method}' not recognized...")
    return None, None


def get_data(n, d, k, seed=42, ktrue=None, noise=0.03, hthresh=0.3):
    rand_state_ = np.random.RandomState(seed)

    if ktrue is None:
        ktrue = k

    # Create the ground truth dataset, with noise
    Wtrue = rand_state_.rand(d, ktrue) 
    Htrue = rand_state_.rand(ktrue, n)
    Htrue /= Htrue.sum(axis=0).reshape(1, n)
    Htrue *= np.maximum(rand_state_.rand(1, n), hthresh)
    N = rand_state_.randn(d, n)*noise
    X = Wtrue @ Htrue + N

    # Now, select a subset of 
    subset = rand_state_.choice(n, k, replace=False)
    W = X[:,subset]

    return X, W


if __name__ == "__main__":
    parser = ArgumentParser(description="Main file for running benchmarking of NMF projection.")
    parser.add_argument("--resultsdir", default="./results", help="Location to save results, if args.save = 1")
    parser.add_argument("--save", default=1, type=int, help="Boolean (i.e., integer 1 or 0) of whether or not to save the results to file.")
    parser.add_argument("--numseeds", default=10, type=int, help="Number of random trials (indexed by seeds) to perform")
    parser.add_argument("--postfix", default="", type=str, help="Postfix identifier string to be differentiate this run from others")
    parser.add_argument("--noise", default=0.03, type=float, help="Noise std to be added to the dataset")
    parser.add_argument("--maxiter", default=5000, type=int, help="Max # iters for the projection alrgorithm")
    args = parser.parse_args()
    
    
    Ns = np.logspace(2, 8, 21).astype(int)
    Ds = np.linspace(2, 200, 21).astype(int)
    seeds = np.arange(42, 42+args.numseeds)

    results = {}
    for method in ['ogm', 'fpgm']:
        print(f"------------ Running Benchmarks for {method} ----------------")
        method_results = {}
        NDS = list(product(Ns, Ds))
        for n, d in tqdm(NDS, total=len(NDS)):
            #print(f"\tn = {n}, d = {d}, numseeds = {args.numseeds}\n")
            Ks = [k_ for k_ in np.linspace(2, 20, 11).astype(int) if k_ < n]
            for k in Ks:
                iters, eps = [], []
                for seed in seeds:
                    X, W = get_data(n, d, k, seed=seed, noise=args.noise)
                    _, info_dict = run_test(X, W, method, maxiter=args.maxiter, verbose=True)
                    iters.append(info_dict['iters'])
                    eps.append(info_dict['eps'])
                method_results[(n,d,k)] = {'iters': iters, 'eps':eps}
        
        results['method'] = method_results


    if args.save:
        if not os.path.exists(args.resultsdir):
            os.makedirs(args.resultsdir)
        savename = os.path.join(args.resultsdir, "proj_bench_ns" + str(args.numseeds) + "_noise" + str(args.noise) + "_" + args.postfix + ".pkl")
        print(f"Saving results to file {savename}...")
        with open(savename, 'wb') as rfile:
            pickle.dump(results, rfile)
        

