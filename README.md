# adaptive-sampling
Code for adaptive sampling framework.

See ``requirements.txt`` file for some necessary packages for running this code.

The main idea is to be able to run tests via the ``main.py`` file as follows:
```python3 main.py --dataset <DATASET NAME> --k <# of samples> --evalenergy <EVALUATION ENERGY>```
* ``dataset``: the name of the dataset the test is running, loading is handled in the ``load_dataset()`` function in ``main.py`. 
* ``k``: number of sample points (centroids, landmarks) to select
For example, can run ``python3 main.py --dataset salinas --k 10`` to run the adaptive sampling code on the Salinas-A dataset (saved in ``./data/`` directory) to select 10 points.
* ``evalenergy``: Energy type on which  to evaluate the different methods' selected sets (see __Notes__ below) 


## Notes
* We run "methods" of the form ``<ENERGY TYPE>-<POLICY>`` where ``ENERGY TYPE`` refers to the type of energy that the adaptive sampling distribution is based on and ``POLICY`` refers to how the next point is selected (either random or greedy).
* Energy types: 
	* ``kmeans``: Kmeans energy 
	* ``lp``: Lp Subspace energy, currently implemented for p = 2
	* ``lpkernel``: Lp Subspace energy with a kernel, p=2. This is RPCholesky on Kernel Gram matrix.
	* ``cvx``: Convex Hull energy, used for NMF.
* Policy types:
	* ``rand``: Sample proportional to the individual terms (i.e., squared distances for p = 2) at each iteration
	* ``greedy``: Select the maximizer at each iteration (i.e., farthest point sampling with corresponding energy distance)
* Each sampling method corresponds to an energy type (above), but for a given test we may want to evaluate the chosen selected sets on a single kind of energy. For example, for HSI datasets, we may want to evaluate the convex hull energy, even though we may choose points by the ``lp-greedy`` method. This "evaluation energy" is specified as the parameter ``--evalenergy`` in ``main.py``.

