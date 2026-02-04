import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import yaml
from time import perf_counter
from argparse import ArgumentParser

from datasets import *
from util import *


ALL_METHODS = ["search", "sampling", "uniform", "search_search", "sampling_sampling"] #, "sampling_search"] 
OVERSAMPLE_METHODS = ["sampling", "uniform"] 


def prep_args(args):
    overwrite_methods = []
    methods = []
    args.powers = [1, 2, 5, None]

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
        assert np.isin(overwrite_methods, ALL_METHODS).all()  # see ALL_METHODS in util.py
        args.overwrite = overwrite_methods

        if 'methods' in config:
            methods = sorted(config['methods'])
            assert np.isin(methods, ALL_METHODS).all()
        else:
            methods = ALL_METHODS 
        args.methods = methods

        if 'oversample' in config:
            oversample = sorted(config['oversample'])
            assert np.isin(oversample, ALL_METHODS).all()
        else:
            oversample = OVERSAMPLE_METHODS
        args.oversample = oversample
        

        if 'k' in config:
            args.k = config['k']
        if 'k_oversample' in config:
            args.k_oversample = config['k_oversample']
        if 'dataset' in config:
            args.dataset = config['dataset']
        if 'energy' in config:
            args.energy = config['energy']
        if 'numseeds' in config:
            args.numseeds = config['numseeds']
        if 'powers' in config:
            args.powers = config['powers']

    args.seeds = np.arange(42, 42+args.numseeds) # define the random seeds

    assert args.energy in IMPLEMENTED_ENERGIES
    return args


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
    parser.add_argument("--record", default=1, type=int, help="Bool flag (0 or 1) of whether or not to record times and other info for each iteration of methods.")
    parser.add_argument("--njobs", default=12, type=int, help="Number of CPU cores to utilize in parallelization.")
    parser.add_argument("--config", default="", type=str, help="Location of .yml configuration file containing 'methods' list.")
    parser.add_argument("--ntest", default=500, type=int, help="Size of 'test' dataset for timing comparisons.")
    args = parser.parse_args()

    args = prep_args(args)

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

        run_experiments_with_p(args, X, p, labels=labels)
