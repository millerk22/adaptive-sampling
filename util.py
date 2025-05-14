import numpy as np
import os 
import pickle
from tqdm import tqdm
from collections import defaultdict

from energies import *
from sampling import *

ALL_METHODS = ["search", "sampling", "uniform"] #, "search_search", "sampling_search", "sampling_sampling", "search_sampling"]  
METHODS = ALL_METHODS #[:-1]
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

    return already_done, methods_to_do, savename, results



def run_experiment(X, p, method_str, results, seeds, args):
    if len(method_str.split("_")) == 2:
        build_method, swap_method = method_str.split("_")
    else:
        build_method = method_str
        swap_method = None
    
    if build_method in OVERSAMPLE_METHODS: # if this build method is designated to be oversampled, set the value of k accordingly
        k_todo = args.k_oversample 
    else:
        k_todo = args.k

    for i, seed in tqdm(enumerate(seeds), total=len(seeds)):
        results[method_str]["seeds"].append(seed)

        if swap_method is None:
            # Instantiate an Energy object for this test
            if args.energy == "conic":
                Energy = ConicHullEnergy(X, k_todo, p=p, n_jobs=args.njobs)
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
                if build_method == "sampling":
                    sampler = AdaptiveSampling(Energy, seed=seed, report_timing=args.time)
                elif build_method == "search":
                    sampler = AdaptiveSearch(Energy, seed=seed, report_timing=args.time)
                else:
                    raise NotImplementedError(f"build_method = {build_method} not recognized...")

                sampler.build_phase(k_todo)
                if args.time:
                    results[method_str]["times"].append(sampler.times)

            results[method_str]["indices"].append(Energy.indices)
            results[method_str]["energy"].append(Energy.energy)
            results[method_str]["energy_values"].append(Energy.energy_values)
            
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
    
    return results 