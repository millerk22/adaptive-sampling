import numpy as np
import matplotlib.pyplot as plt
import pickle
from datasets import load_dataset
from collections import defaultdict

from energies import * 


linestyles = {'sampling': 'ks-', 'search': 'bo-', 'uniform':'rx-',
              'search_search': 'g.-', 'sampling_sampling':'c--', 'sampling_search':'o--',
              'reference': 'k--'}
methods_all = ['uniform', 'sampling', 'search', 'sampling_sampling', 'search_search', 'sampling_search']
get_idx = {method:i for i, method in enumerate(methods_all)}
colors = ['#CC79A7', '#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#000000']
markers = ['v', 'o', 's', '^',  'D', '*', '']
linestyles = [':', '-.', '-', '-','--', '--', ':']
plotorder = {"sampling_sampling":10, "sampling":8, "search":7, "search_search":9, "uniform":2, "sampling_seearch":1}
names = ["-".join(method.split("_")) for method in methods_all]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],  # or "Computer Modern Roman"
    "axes.labelsize": 15,
    "font.size": 15,
    "legend.fontsize": 13,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})


def get_k_by_energy(results, n, p=2):
    data = {}
    divisor = 1.0
    if p is not None:
        divisor = n**(1.0/float(p))

    for method, res in results.items():
        if len(method.split("_")) > 1: # if is swap method
            data[method] = np.array(res['swap_values']) / divisor
        else:
            data[method] = np.array(res['build_values']) / divisor
    return data

def get_num_forced_trivial_swaps(samp_samp_res):
    forced_swaps_counts = defaultdict(list)
    trivial_swaps_counts = defaultdict(list)
    tot_num_swaps = {}
    for w_dict in samp_samp_res['all_swap_stag']:
        for k, w in w_dict.items():
            forced_swaps_counts[k].append((np.array(w) >= k).sum())
            trivial_swaps_counts[k].append(np.sum(w))
            tot_num_swaps[k] = len(w)
    # returns dictionary with key : value
    #    - key = k
    #    - value = list of number of forced swaps (length of list is number of seeds) 
    return forced_swaps_counts, trivial_swaps_counts, tot_num_swaps



def visualize_chosen_images(results, datasetname, k=0, p=2, ncols=5, figsize=(10,10), cmap="gray", imgshape=(25,25)):
    X, labels = load_dataset(datasetname)
    print(np.unique(labels))
    try:
        X = np.array([X[:,i].reshape((imgshape[0], imgshape[1])) for i in range(X.shape[1])]) # shape (height, width)
    except:
        raise ValueError(f"image shape {imgshape} not compatible with dataset")
    print(X.shape)
    for method, res in results.items():
        if method == "sampling_search":
            continue
        if method == "uniform":
            continue
        print(f"---------------{method}---------------")
        
        prototypes = res['indices'][-1]
        if type(prototypes) is list:
            if k > 0:
                prototypes = prototypes[:k]
            else:
                k = len(prototypes)
        else:
            if k > 0:
                prototypes = prototypes[k][:k]
            else:
                prototypes = prototypes[max(list(prototypes.keys()))][:]
                k = len(prototypes)
        
        images = X[prototypes]
        titles = labels[prototypes]
        ind_sort = np.argsort(titles)
        images = images[ind_sort]
        titles = titles[ind_sort]
        nrows = (k + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
        axes = axes.ravel()
        for j, ax in enumerate(axes.flat):
            ax.imshow(images[j], cmap="gray")
            ax.set_title("Label = {}".format(titles[j]), fontsize=9)
            ax.axis("off")
        plt.show()
    return 


def get_prototypes(results, run_num=0):
    prototypes = {}
    for method, res in results.items():
        is_build = len(method.split("_")) == 1
        if is_build:
            prototypes[method] = res["indices"][0]
        else:
            all_prototypes = res["indices"][0]
            prototypes[method] = all_prototypes[max(all_prototypes.keys())]
    return prototypes


def get_reconstructions(results, X, kind, kvalues=[5, 10, 15, 20], p=2):
    assert p is not None
    reconstructions = defaultdict(list)

    ## Get the average reconstruction error for search_search to get a representative sample to visualize for every method and k value
    prototypes = results['search_search']['indices'][-1] # consider the last seed's trial
    if type(prototypes) is dict:
        prototypes = prototypes[max(list(prototypes.keys()))]
    kvalues_done = []
    for k in kvalues:
        try:
            if kind == "convex":
                energy = ConvexHullEnergy(X, p=p)
            elif kind == "conic":
                energy = ConicHullEnergy(X, p=p)    
            elif kind == "lowrank":
                energy = LowRankEnergy(X, p=p)
            else:
                raise NotImplementedError()
            energy.init_set(prototypes[:k])
        except:
            print(f"Could not compute reconstruction for method {method} with k={k}")
            print("\tlen(prototypes) =", len(prototypes), ", but kvalues = ", kvalues)
            print("\tskipping...")
            continue 
        
        kvalues_done.append(k)

        if kind in ["convex", "conic"]:
            X_recon = energy.W @ energy.H
        elif kind == "lowrank":
            X_recon = energy.X[:, energy.indices] @ energy.W

        recon_errs = np.linalg.norm(X - X_recon, axis=0, ord=p) / X.shape[0]
        avg_idx = np.argmin(np.absolute(recon_errs - np.mean(recon_errs))) # find the idx of the sample with reconstruction error closest to the average
        reconstructions["search_search"].append((X_recon[:,avg_idx], recon_errs[avg_idx]))
        reconstructions["original"].append((X[:,avg_idx], 0.0))
    

    for method, res in results.items():
        if method == "search_search":
            continue 

        prototypes = res['indices'][-1] # consider the last seed's trial
        if type(prototypes) is dict:
            prototypes = prototypes[max(list(prototypes.keys()))]

        for k in kvalues_done:
            if kind == "convex":
                energy = ConvexHullEnergy(X, p=p)
            elif kind == "conic":
                energy = ConicHullEnergy(X, p=p)    
            elif kind == "lowrank":
                energy = LowRankEnergy(X, p=p)
            else:
                raise NotImplementedError()
            energy.init_set(prototypes[:k])
                
            if kind in ["convex", "conic"]:
                X_recon = energy.W @ energy.H
            elif kind == "lowrank":
                X_recon = energy.X[:, energy.indices] @ energy.W

            recon_errs = np.linalg.norm(X - X_recon, axis=0, ord=p) / X.shape[0]
            reconstructions[method].append((X_recon[:,avg_idx], recon_errs[avg_idx]))
        
    return reconstructions, kvalues_done




def get_time_by_energy(results, n, k, p=2):
    data = {}
    divisor = 1.0
    if p is not None:
        divisor = n**(1.0/float(p))

    for method in sorted(results.keys()):
        if method == "uniform":
            continue
        res = results[method]
        if len(method.split("_")) == 1:
            build_vals = np.array(res["build_values"])/divisor
            data[method] = (build_vals, np.cumsum(np.array(res["build_times"]), axis=1))
        else:
            swap_vals = [np.array(vals[k])/divisor for vals in res["all_swap_values"]]
            swap_times = [np.array(vals[k]) for vals in res["all_swap_times"]]
            swap_vals = [np.concatenate((data[method.split("_")[0]][0][i,:k], swap_vals[i])) for i in range(len(swap_vals))] # append the build values 
            if method.split("_")[-1] == "sampling":
                swap_vals = [np.minimum.accumulate(swap_vals[i]) for i in range(len(swap_vals))]
            build_times = data[method.split("_")[0]][1][:,:k]
            swap_times = [np.concatenate((build_times[i,:], build_times[i,-1] + np.cumsum(swap_times[i]))) for i in range(len(swap_times))]
            data[method] = (swap_vals, swap_times)
    return data 

def get_calls_by_energy(results, n, k, p=2):
    data = {}
    divisor = 1.0
    if p is not None:
        divisor = n**(1.0/float(p))

    for method in sorted(results.keys()):
        if method == "uniform":
            continue
        res = results[method]
        if len(method.split("_")) == 1:
            build_vals = np.array(res["build_values"])/divisor
            if method == "sampling":
                build_calls = np.array([np.arange(1, build_vals.shape[1]+1) for s in range(build_vals.shape[0])])
            elif method == "search":
                build_calls = np.array([np.cumsum(np.array([n-j for j in range(k)])) for s in range(build_vals.shape[0])])
            else:
                raise ValueError
            data[method] = (build_vals, np.cumsum(build_calls, axis=1))
        else:
            swap_vals = [np.array(vals[k])/divisor for vals in res["all_swap_values"]]
            swap_stags = [vals[k] for vals in res["all_swap_stag"]]
            
            swap_vals = [np.concatenate((data[method.split("_")[0]][0][i,:k], swap_vals[i])) for i in range(len(swap_vals))] # append the build values 
            if method.split("_")[-1] == "sampling":
                swap_vals = [np.minimum.accumulate(swap_vals[i]) for i in range(len(swap_vals))]
                swap_calls = [[2*(s+1) for s in stag] for stag in swap_stags]
            elif method.split("_")[-1] == "search":
                swap_calls = [[k*(s+1) for s in stag] for stag in swap_stags]
            
            build_calls = data[method.split("_")[0]][1][:,:k]
            swap_calls = [np.concatenate((build_calls[i,:] , build_calls[i,-1]+np.cumsum(swap_calls[i]))) for i in range(build_calls.shape[0])]
            data[method] = (swap_vals, swap_calls)
    return data 


def is_mono(l):
    bools = [l[i+1] <= l[i] for i in range(len(l)-1)]
    return bools