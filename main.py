import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import yaml
from argparse import ArgumentParser

from datasets import *
from util import *


def run_experiments_p(args, X, p, seeds=np.arange(42,48), overwrite=[]):

    already_done, methods_to_do, savename, results = find_methods_to_do(args, p, overwrite)    
    
    print("Already found results for: ", ", ".join(already_done))
    print("(Re-)Computing results for: ", ", ".join(methods_to_do))
    print(f"\toversample methods = {OVERSAMPLE_METHODS}")
    print("="*40)
    
    for count, method_str in enumerate(methods_to_do):
        if len(results[method_str]['energy']) == args.numseeds:
            print(f"Already found {method_str} result in {savename}, skipping...")
            continue

        print(f"Overall Method = {method_str}, {count+1}/{len(methods_to_do)}")
        
        results = run_experiment(X, p, method_str, results, seeds, args)

        if args.save:
            print(f"Saving (intermediate) results to file {savename}...")
            with open(savename, 'wb') as rfile:
                pickle.dump(results, rfile)
    return 





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

    # check config file
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

    assert args.energy in IMPLEMENTED_ENERGIES

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

        run_experiments_p(args, X, p, seeds=np.arange(42,42+args.numseeds), overwrite=overwrite_methods)