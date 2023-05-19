# p-value Adjustment for Monotonous, Unbiased, and Fast Clustering Comparison

This repository contains the code to reproduce the experiments in our paper *$p$-value Adjustment for Monotonous, Unbiased, and Fast Clustering Comparison*.

Install the dependencies with `conda`:
```
conda env create -f ./environment.yml
```

Compile our cython implementation of the standardized mutual information:
```
python setup.py build_ext --inplace
```