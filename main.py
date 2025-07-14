import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import yaml
from time import perf_counter
from argparse import ArgumentParser

from datasets import *
from util import *


def run_experiments_p(args, X, p, labels=None, seeds=np.arange(42,48), overwrite=[]):

    already_done, methods_to_do, savename, results = find_methods_to_do(args, p, overwrite)    
    
    print("Already found results for: ", ", ".join(already_done))
    print("(Re-)Computing results for: ", ", ".join(methods_to_do))
    print("\tOversample methods: ", ", ".join(OVERSAMPLE_METHODS))
    print("\tOverwrite methods: ", ", ".join(overwrite))

    compute_reference = False
    if "reference_energy" not in results and labels is not None:
        if args.energy.split("-")[0] == "cluster": # only will compute reference energy for the cluster
            compute_reference = True 
            print("\tNeed to compute reference energy...")
    print("="*60)

    if compute_reference:
        print("\nComputing reference energy...")
        tic = perf_counter()
        reference_inds = get_reference_inds(X.T, labels, p=p)
        if args.energy == "cluster":
            energy = ClusteringEnergy(X, p=p)
        elif args.energy == "cluster-dense":
            energy = ClusteringEnergyDense(X, p=p)
        else:
            raise NotImplementedError(f"Computing reference_energy for energy = '{args.energy}' not currently implemented")
        
        energy.init_set(reference_inds)
        sampler = AdaptiveSampler(energy)
        sampler.swap_phase("search", max_swaps=500)

        reference_k, reference_energy = np.unique(labels).size, energy.energy
        results['reference'] = [reference_k, reference_energy]
        toc = perf_counter()
        print(f"\ttime to compute = {toc -tic}\n")
        with open(savename, 'wb') as rfile:
            pickle.dump(results, rfile)

    
    for count, method_str in enumerate(methods_to_do):
        if len(results[method_str]['energy']) == args.numseeds:
            print(f"Already found {method_str} result in {savename}, skipping...")
            continue

        print(f"Overall Method = {method_str}, {count+1}/{len(methods_to_do)}")
        
        results = run_experiment(X, p, labels, method_str, results, seeds, args)

        if args.save:
            print(f"Saving (intermediate) results to file {savename}...")
            with open(savename, 'wb') as rfile:
                pickle.dump(results, rfile)
    return 





if __name__ == "__main__":
    parser = ArgumentParser(description="Main file for running adaptive search/sampling tests.")
    parser.add_argument("--dataset", default="test", help="Name of dataset, referenced in get_dataset function")
    parser.add_argument("--k", default=10, type=int, help="Number of samples to select")
    parser.add_argument("--k_oversample", default=50, type=int, help="Number of samples to select via oversampling")
    parser.add_argument("--resultsdir", default="./results", help="Location to save results, if args.save = 1")
    parser.add_argument("--save", default=1, type=int, help="Boolean (i.e., integer 1 or 0) of whether or not to save the results to file.")
    parser.add_argument("--energy", default="conic", help="String name of the evaluation energy type; one of ['conic', 'cluster']")
    parser.add_argument("--numseeds", default=1, type=int, help="Number of random trials (indexed by seeds) to perform")
    parser.add_argument("--postfix", default="", type=str, help="Postfix identifier string to be differentiate this run from others")
    parser.add_argument("--time", default=1, type=int, help="Bool flag (0 or 1) of whether or not to record times for each iteration of methods.")
    parser.add_argument("--njobs", default=12, type=int, help="Number of CPU cores to utilize in parallelization.")
    parser.add_argument("--config", default="", type=str, help="Location of .yml configuration file containing 'methods' list.")
    parser.add_argument("--ntest", default=500, type=int, help="Size of 'test' dataset for timing comparisons.")
    args = parser.parse_args()

    
    overwrite_methods = []
    args.powers = [1, 2, None]

    if args.dataset == "test":
        overwrite_methods = [] 
        args.powers = [2, None]

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
    X, labels = load_dataset(args.dataset, n_test=args.ntest)

    if args.dataset == "test":
        args.dataset = "test" + str(args.ntest)
    
    # run the test
    print(f"------------ Running Test for {args.dataset} ----------------")
    print(f"\tn = {X.shape[1]}, k = {args.k}, k_oversample = {args.k_oversample}, numseeds = {args.numseeds}, njobs = {args.njobs}")
    
    for p in args.powers:
        print()
        print("="*60)
        print(f"================== p = {p}, energy_type = {args.energy} =======================")
        print("="*60)
        if p == "None":
            p = None

        run_experiments_p(args, X, p, labels=labels, seeds=np.arange(42,42+args.numseeds), overwrite=overwrite_methods)
