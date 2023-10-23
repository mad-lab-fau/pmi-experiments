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

# Clustering experiments
python real_experiment_clustering.py --download
# Figure 4a and 4b
python real_experiment_clustering.py -r 1000 -o ./results/clustering.csv -d olivetti digits -a kmeans
# Appendix E Figure 5
python real_experiment_clustering.py -r 1000 -s -o ./results/clustering_spectral.csv -d segment texture -a spectral

# Community detection experiment in Figure 4c
python real_experiment_community_detection.py -c 30 -r 100

# Examples for type II bias in Table 2
python test_biases.py
```