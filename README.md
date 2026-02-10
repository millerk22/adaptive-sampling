# adaptive-sampling

Code for adaptive sampling framework. See the ``demo.ipynb`` for a brief demo of usage of the ``Energy(X, k=k, p=p)`` and ``AdaptiveAlgorithm(Energy)`` objects for specific dataset applications. 

See ``requirements.txt`` file for some necessary packages for running this code.

---

### Energy Types: 
* ``cluster``: $A(\mathcal{Y}) = \mathcal{Y}$. Relates to kmedoids and kmeans clustering.
* ``lowrank``: $\mathcal{A}(\mathcal{Y}) = \text{span}(\mathcal{Y})$. Relates to low rank matrix factorization.
* ``conic``: $\mathcal{A}(\mathcal{Y}) = \text{coni}(\mathcal{Y}) ( = \text{conv}(\mathcal{Y}\cup \{\mathbf{0}\}))$. Relates to nonnegative matrix factorization (NMF).
* ``convex``: $\mathcal{A}(\mathcal{Y}) = \text{conv}(\mathcal{Y})$. Relates to archetypal/archetypoid analysis.
 


# Build vs Swap Phases
Constructing a prototype set is split into 2 phases: __build__ and __swap__.
* In the __build__ phase, we iteratively construct the prototype set with a build method (i.e., sampling build or search build)
* In the __swap__ phase, we iteratively refine an already chosen prototype set with a swap method (i.e., sampling swap or search swap)

### Build Methods:
* ``uniform``: Sample a subset uniformly at random from the dataset (non-adaptive) 
* ``sampling``: Adaptive Sampling Build with power ``p``. At each step, sample a new prototype $x_i$ with probability proportional to the individual distances ``d(x_i, S)**p`` (i.e., squared distances for ``p = 2``) at each iteration.
* ``search``: Adaptive Search Build with power ``p``. At each step, induct whichever new prototype leads to the maximum decrease in the cost function.

### Swap Methods:
* ``sampling``: Adaptive Sampling Swap with power ``p``. Cyclically iterate through the prototype set, evict the selected prototype, and resample a new prototype $x_i$ with probability proportional to ``d(x_i, S)``. 
* ``search``: Adaptive Search Swap with power ``p``, with "eager swapping". Cyclically iterate through each data point not in the prototype set, add it to the prototype set, and remove whichever prototype leads to the smallest increase in the cost function.


--- 

## Using ``main.py`` for tests

The idea is to be able to run many tests directly in a command line/terminal via the ``main.py`` Python script as follows:
```python3 main.py --dataset <DATASET NAME> --k <# of samples> --energy <EVALUATION ENERGY>```
* ``dataset``: the name of the dataset the test is running, loading is handled in the ``load_dataset()`` function found in ``datasets.py`. 
* ``k``: number of sample points (i.e., prototypes, centroids, landmarks) to select
For example, can run ``python3 main.py --dataset salinas --k 10`` to run the adaptive sampling code on the Salinas-A dataset (saved in ``./data/`` directory) to select 10 points.
* ``energy``: Energy type on which to evaluate the different methods' selected sets 

#### Use of config files

The script ``main.py`` also allows for passing in the path to a config ``.yml`` file that specifies experiment parameter settings. See ``./config.yml`` for an example file format. Usage through the command line is simply:

``python3 main.py --config config.yml``