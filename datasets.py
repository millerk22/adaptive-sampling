import numpy as np
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import make_blobs, load_iris, load_digits, load_wine, load_breast_cancer
from graphlearning.datasets import load
from graphlearning.trainsets import generate
from graphlearning import weightmatrix as wm
from graphlearning import graph

def load_dataset(dataset_name, n_test=500):
    if dataset_name == "blobssmallest":
        X, labels = make_blobs(5*[50], n_features=200, cluster_std=0.1, random_state=42)
    elif dataset_name == "blobssmall":
        X, labels = make_blobs(5*[100], n_features=200, cluster_std=0.1, random_state=42)
    elif dataset_name == "blobs":
        X, labels = make_blobs(5*[500], n_features=20, cluster_std=0.01, random_state=42)
    elif dataset_name == "blobs2":
        X, labels = make_blobs(5*[50], n_features=2, cluster_std=0.1, random_state=42)
    elif dataset_name == "blobs5":
        X, labels = make_blobs(5*[50], n_features=5, cluster_std=0.1, random_state=42)
    elif dataset_name == "blobs20":
        X, labels = make_blobs(5*[50], n_features=20, cluster_std=0.1, random_state=42)
    elif dataset_name == "blobs200":
        X, labels = make_blobs(5*[50], n_features=200, cluster_std=0.01, random_state=42)
    elif dataset_name == "apartment":
        data = fetch_ucirepo(id=555)
        X = data.data.features
        labels = data.data.targets
        print(X.shape)
    elif dataset_name == "news":
        data = fetch_ucirepo(id=332)
        X = data.data.features
        labels = data.data.targets
        print(X.shape)
    elif dataset_name == "adult":
        data = fetch_ucirepo(id=20)
        X = data.data.features
        labels = data.data.targets
        print(X.shape)
    elif dataset_name == "yalefaces":
        X, labels = np.load("./data/yalefaces.npz", allow_pickle=True).values()
        print(X.shape)
    elif dataset_name == "iris":
        X, labels = load_iris(return_X_y=True)
    elif dataset_name == "digits":
        X, labels = load_digits(return_X_y=True)
    elif dataset_name == "wine":
        X, labels = load_wine(return_X_y=True)
    elif dataset_name == "breastcancer":
        X, labels = load_breast_cancer(return_X_y=True)
    elif dataset_name == "urban":
        data = np.load("./data/urban.npz")
        X = data['H']
        X = 1.0*X
        X -= min(0.0, X.min())
        X /= np.max(np.max(X))
        X /= np.linalg.norm(X, axis=1).reshape(-1, 1)
        labels = data['labels']

    elif dataset_name == "urbansub":
        data = np.load("./data/urban.npz")
        X = data['H']
        labels = data['labels']
        X = 1.0*X
        X -= min(0.0, X.min())
        X /= np.max(np.max(X))
        X /= np.linalg.norm(X, axis=1).reshape(-1, 1)
        rstate = np.random.RandomState(42)
        subset = rstate.choice(X.shape[0], 7500, replace=False)
        print(X.shape)
        X = X[subset]
        print(X.shape)
        labels = labels[subset]

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
        X /= np.linalg.norm(X, axis=1).reshape(-1, 1)
        labels = labels[mask]
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
        X /= np.linalg.norm(X, axis=1).reshape(-1, 1)
        print(X.shape)
        labels = labels[mask]
    
    elif dataset_name == "paviasub":   # HSI dataset
        X = np.load("./data/pavia.npz")['H']
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        labels = np.load("./data/pavia.npz")['labels']
        labels = np.reshape(labels, (labels.shape[0]*labels.shape[1], 1)).flatten()
        mask = labels != 0
        X = X[mask]
        labels = labels[mask]
        X = 1.0 * X
        X -= min(0.0, X.min())
        X /= np.max(np.max(X))
        X /= np.linalg.norm(X, axis=1).reshape(-1, 1)
        rstate = np.random.RandomState(42)
        subset = rstate.choice(X.shape[0], 5000, replace=False)
        print(X.shape)
        X = X[subset]
        print(X.shape)
        labels = labels[subset]
    
    elif dataset_name == "gaussian":
        rand_state = np.random.RandomState(42)  
        X = rand_state.randn(500, 50)
        labels = None 
    
    elif dataset_name == "snp":
        labels = np.load("./data/snps/labels.npy", allow_pickle=True)
        X = np.load("./data/snps/data.npy", allow_pickle=True)
        ordering = np.argsort(labels)
        X, labels = X[ordering], labels[ordering]
        nan_inds = np.where(np.isnan(X))
        col_means = np.nanmean(X, axis=0)
        X[nan_inds] = np.take(col_means, nan_inds[1])
        X = X.T.astype(np.float64)
        print(X.shape, np.unique(labels).size)

    elif dataset_name == "smile":
        X, bw = smile(10000)
        labels = None
        print(X.shape)
    elif dataset_name == "outliers":
        X, _ = outliers(10000)
        labels = None 
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
        labels = None 
    elif dataset_name == "mnist":
        X, labels = load("mnist", metric="vae")
        inds = generate(labels, rate=1000, seed=42)
        X = X[inds]
        labels = labels[inds]
    elif dataset_name == "mnistsc":
        X, labels = load("mnist", metric="raw")
        W = wm.knn(X, k=30)
        G = graph(W)
        print(f"Warning... Computing eigenvectors for {G.num_nodes}-nodes graph")
        _, evecs = G.eigen_decomp(k=15, method='lowrank')
        X_ = evecs[:,1:]
        inds = generate(labels, rate=1000, seed=42)
        X = X_[inds]
        labels = labels[inds]
    elif dataset_name == "mnistraw":
        X, labels = load("mnist", metric="raw")
        inds = generate(labels, rate=1000, seed=42)
        X = X[inds]
        labels = labels[inds]
    elif dataset_name == "cifar10":
        X, labels = load("cifar", metric="simclr")
        inds = generate(labels, rate=1000, seed=42)
        X = X[inds]
        labels = labels[inds]
    elif dataset_name == "cifar10sc":
        X, labels = load("cifar", metric="raw")
        W = wm.knn(X, k=30)
        G = graph(W)
        print(f"Warning... Computing eigenvectors for {G.num_nodes}-nodes graph")
        _, evecs = G.eigen_decomp(k=15, method='lowrank')
        X_ = evecs[:,1:]
        inds = generate(labels, rate=1000, seed=42)
        X = X_[inds]
        labels = labels[inds]
    elif dataset_name == "cifar10raw":
        X, labels = load("cifar", metric="raw")
        inds = generate(labels, rate=1000, seed=42)
        X = X[inds]
        labels = labels[inds]
    else:
        raise ValueError(f"Dataset = {dataset_name} not recognized")
    
    return X.T, labels



def smile(N, bandwidth = 2.0, **kwargs):
    small = int(np.ceil(N ** (1.0/2)))
    eye_points = kwargs["eye_points"] if ("eye_points" in kwargs) else small
    mouth_points = kwargs["mouth_points"] if ("mouth_points" in kwargs) else int(np.ceil(N/10.0))
    face_points = N - 2 * eye_points - mouth_points

    X = np.zeros((N, 2))
    idx = 0

    # Eyes
    for x_shift in [-4.0, 4.0]:
        for i in range(eye_points):
            while True:
                x = 2 * np.random.rand() - 1
                y = 2 * np.random.rand() - 1
                if x**2 + y**2 <= 1.0:
                    X[idx, 0] = x + x_shift
                    X[idx, 1] = y + 4.0
                    idx += 1
                    break

    # Mouth
    for x in list(np.linspace(-5.0, 5.0, mouth_points)):
        X[idx, 0] = x
        X[idx, 1] = x**2 / 16.0 - 5.0
        idx += 1

    # Face
    for theta in list(np.linspace(0, 2*np.pi, face_points)):
        X[idx, 0] = 10.0 * np.cos(theta)
        X[idx, 1] = 10.0 * np.sin(theta)
        idx += 1

    return X, bandwidth

def robspiral(N):
    times = np.linspace(0, 2, N)
    times = times ** 6
    times = times[::-1]
    x = np.exp(.2 * times) * np.cos(times)
    y = np.exp(.2 * times) * np.sin(times)
    X = np.column_stack((x,y))
    bandwidth = 1000
    return X, bandwidth


def outliers(N, num_outliers = 50):
    X = 0.5*np.random.randn(N, 20)/np.sqrt(20.0)
    X[np.random.choice(range(N), size = num_outliers, replace = False),:] += 100.0 * np.random.randn(num_outliers, 20)
    return X, None
