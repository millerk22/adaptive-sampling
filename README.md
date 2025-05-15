# adaptive-sampling

Code for adaptive sampling framework. See the ``demo.ipynb`` for a brief demo of usage of the ``Energy(X, k=k, p=p)`` and ``AdaptiveSampler(Energy)`` objects for specific dataset applications. 

See ``requirements.txt`` file for some necessary packages for running this code.

---

## Using ``main.py`` for tests

The idea is to be able to run many tests via the ``main.py`` file as follows:
```python3 main.py --dataset <DATASET NAME> --k <# of samples> --energy <EVALUATION ENERGY>```
* ``dataset``: the name of the dataset the test is running, loading is handled in the ``load_dataset()`` function found in ``datasets.py`. 
* ``k``: number of sample points (i.e., prototypes, centroids, landmarks) to select
For example, can run ``python3 main.py --dataset salinas --k 10`` to run the adaptive sampling code on the Salinas-A dataset (saved in ``./data/`` directory) to select 10 points.
* ``energy``: Energy type on which to evaluate the different methods' selected sets 

### Energy Types: 
* ``conic``: Conic Hull energy, used for NMF.
* ``clustering``: __(not currently implemented, in progress)__


# Build vs Swap Phases
We split the process into 2 phases: __build__ and __swap__.
* In the __build__ phase, we iteratively construct the prototype set with a build method (i.e., adaptive sampling build or adaptive search build)
* In the __swap__ phase, we iteratively refine an already chosen prototype set with a swap method (i.e., adaptive sampling swap or adaptive search swap)

### Build Methods:
* ``uniform``: Sample a subset uniformly at random from the dataset (non-adaptive) 
* ``sampling``: Adaptive Sampling Build with power ``p``. Sample proportional to the individual distances ``d(x_i, S)**p`` (i.e., squared distances for ``p = 2``) at each iteration.
* ``search``: Adaptive Search Build with power ``p``. Sample points that maximally decrease the objective function ("energy") the most at each iteration. 

### Swap Methods:
* ``sampling``: Adaptive Sampling Swap with power ``p``. Cycle through the current prototype set's indices, eject and sample a replacement according to adaptive sampling distribution (i.e., proportional to ``d(x_i, S)``)
* ``search``: Adaptive Search Swap with power ``p``. Iteratively find the best possible replacement point (outside the prototype set) to swap with points in the prototype set. Repeat until no possible swaps lead to lower energy (objective function value).
