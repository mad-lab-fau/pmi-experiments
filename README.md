# p-value Adjustment for Monotonous, Unbiased, and Fast Clustering Comparison

This repository contains the code to reproduce the experiments in our paper *$p$-value Adjustment for Monotonous, Unbiased, and Fast Clustering Comparison*.

Install the dependencies with `conda`:
```
conda env create -f ./environment.yml
```

Compile our Cython implementation of the standardized mutual information:
```
python setup.py build_ext --inplace
```

The Monte Carlo and normal approximation to the PMI are implemented in `clustering_comparison_measures.py`. The experiments can be run using:
```
# Synthetic experiments in Figures 1, 2, 3
python synthetic_experiments.py

# Clustering experiments in Figure 4a and 4b
python real_experiment_clustering.py -d 
python real_experiment_clustering.py -r 1000

# Community detection experiment in Figure 4c
python real_experiment_community_detection.py -c 30 -r 100
```