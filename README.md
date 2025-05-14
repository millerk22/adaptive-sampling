# adaptive-sampling

## NEED TO UPDATE README
Code for adaptive sampling framework.

See ``requirements.txt`` file for some necessary packages for running this code.

The main idea is to be able to run tests via the ``main.py`` file as follows:
```python3 main.py --dataset <DATASET NAME> --k <# of samples> --energy <EVALUATION ENERGY>```
* ``dataset``: the name of the dataset the test is running, loading is handled in the ``load_dataset()`` function in ``main.py`. 
* ``k``: number of sample points (centroids, landmarks) to select
For example, can run ``python3 main.py --dataset salinas --k 10`` to run the adaptive sampling code on the Salinas-A dataset (saved in ``./data/`` directory) to select 10 points.
* ``energy``: Energy type on which to evaluate the different methods' selected sets (see __Notes__ below) 


## Notes
* Energy types: 
	* ``kmeans``: Kmeans energy 
	* ``lp``: Lp Subspace energy, currently implemented for p = 2
	* ``lpkernel``: Lp Subspace energy with a kernel, p=2. This is RPCholesky on Kernel Gram matrix.
	* ``cvx``: Convex Hull energy, used for NMF.
* Method types:
    * ``passive``: Sample a subset uniformly at random from the dataset (non-adaptive) 
	* ``rand``: Sample proportional to the individual terms (i.e., squared distances for p = 2) at each iteration
	* ``greedy``: Select the maximizer at each iteration (i.e., farthest point sampling with corresponding energy distance)
