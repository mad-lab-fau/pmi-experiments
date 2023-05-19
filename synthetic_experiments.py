import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import binom
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Union, List, Callable, Tuple, Dict, Any, Optional
from numpy.random import default_rng, SeedSequence, BitGenerator, Generator
from itertools import chain
from inspect import signature
from sklearn.metrics.cluster import contingency_matrix, rand_score
from clustering_comparison_measures import p_value_adjusted_mutual_information_2_normal
import scipy.sparse as sp

# Benchmarking imports
from random_partition import RandomSetPartition
from time import perf_counter
from tqdm import tqdm
import signal


class Timeout:
    def __init__(self, seconds=300, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def _worker_init_fn(metrics: List[Tuple[str, Callable, Dict[str, Any]]]) -> None:
    """Initialize the worker process with a random partition instance and the comparison metrics."""
    global set_partition_generator
    global comparison_metrics
    comparison_metrics = metrics
    set_partition_generator = RandomSetPartition()

def _get_kwargs(labels_true: ArrayLike, labels_pred: ArrayLike, contingency: Optional[sp.csc_matrix], seed: Union[None, SeedSequence, BitGenerator, Generator], kwargs: Dict[str, Any], function: Callable) -> Dict[str, Any]:
    """Get the keyword arguments for a function from a dictionary of keyword arguments.

    Args:
        labels_true (int array[n_samples]):
            A clustering of the data into disjoint subsets.
        labels_pred (int array[n_samples]):
            Another clustering of the data into disjoint subsets.
        contingency (optional):
            The contingency matrix of the two clusterings.
        seed (optional):
            Random seed.
        kwargs (dict):
            The dictionary of keyword arguments.
        function (callable):
            The function to get the keyword arguments for.

    Returns:
        The keyword arguments for the function.
    """
    signature_keys = signature(function).parameters.keys()

    kwargs = kwargs.copy()
    kwargs['labels_true'] = labels_true
    kwargs['labels_pred'] = labels_pred
    kwargs['contingency'] = contingency
    kwargs['seed'] = seed

    return {filter_key: kwargs[filter_key] for filter_key in signature_keys}

def _romano_run_trial(args):
    global set_partition_generator
    global comparison_metrics

    number_datapoints, reference_number_clusters, compare_number_clusters, seed = args

    # Generate reference clustering
    reference_labels = []
    for cluster in range(reference_number_clusters):
        reference_labels.extend(
            [cluster] * (number_datapoints // reference_number_clusters))
    prng = default_rng(seed)
    set_partition_generator.reseed(prng)
    prng.shuffle(reference_labels)

    # Generate clusterings to compare to reference clustering
    comparison_labels = [set_partition_generator.random_partition(
        number_datapoints, number_clusters) for number_clusters in compare_number_clusters]

    results = []
    for metric_name, metric, kwargs in comparison_metrics:
        # Compute metrics for all clusterings
        metric_values = [metric(**_get_kwargs(reference_labels, comparison_label, None, prng, kwargs, metric))
                         for comparison_label in comparison_labels]
        selected_number_of_clusters = compare_number_clusters[np.argmax(
            metric_values)]
        results.append(
            (metric_name, selected_number_of_clusters, metric_values))
    return results


def romano_experiment(comparison_metrics: List[Tuple[str, Callable, Dict[str, Any]]], number_datapoints: int = 500, reference_number_clusters: int = 10, compare_number_clusters: List[int] = [2, 6, 10, 14, 18, 22], trials: int = 5_000, seed: SeedSequence = SeedSequence(), n_jobs: Union[None, int] = None) -> pd.DataFrame:
    """ Run the experiment performed in S. Romano et al. "Standardized Mutual Information
    for Clustering Comparisons: One Step Further in Adjustment for Chance" (2014).

    We generate a reference clustering with reference_number_clusters evenly sized clusters.
    Then we generate a number of clusterings with compare_number_clusters clusters and
    for every comparison metric we record the selected cluster's size. After trials trials
    we return the selection probability for each cluster size for each comparison metric.

    Args:
        comparison_metrics: List of tuples of metric name, metric function and kwargs except labels_true, labels_pred and seed
        number_datapoints: Number of datapoints N
        reference_number_clusters: Number of clusters K
        compare_number_clusters: List of number of clusters to compare to the reference clustering
        trials: Number of trials to run
        seed: random seed sequence to use for reproducibility
        n_jobs: Number of jobs to run in parallel. If None, use all available cores.

    Returns:
        Dataframe with raw results
    """
    if number_datapoints % reference_number_clusters != 0:
        raise ValueError(
            "Number of datapoints must be divisible by number of clusters")

    if n_jobs is None:
        n_jobs = cpu_count()

    params = ((number_datapoints, reference_number_clusters,
              compare_number_clusters, seed.spawn(1)[0]) for _ in range(trials))

    with Pool(n_jobs, initializer=_worker_init_fn, initargs=(comparison_metrics, )) as pool:
        results = list(tqdm(chain.from_iterable(pool.imap_unordered(
            _romano_run_trial, params, chunksize=max(1, trials // (100 * n_jobs)))), total=trials * len(comparison_metrics)))

    df = pd.DataFrame(results, columns=[
                      "metric_name", "number_clusters", "metric_values"])
    return df

if __name__ == "__main__":
    metrics = [
        ("PMI2_normal", p_value_adjusted_mutual_information_2_normal, {}),
        ("RI", rand_score, {}),
    ]

    df = romano_experiment(metrics, number_datapoints=100, reference_number_clusters=10, compare_number_clusters=[2, 6, 10, 14, 18, 22], trials=20, seed=SeedSequence(0), n_jobs=1)
    print(df)