import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import binom
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Union, List, Callable, Tuple, Dict, Any, Optional
from numpy.random import default_rng, SeedSequence, BitGenerator, Generator
from itertools import chain, combinations_with_replacement
from inspect import signature
from sklearn.metrics.cluster import (
    contingency_matrix,
    rand_score,
    entropy,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    mutual_info_score,
)
from clustering_comparison_measures import (
    standardized_rand_score,
    standardized_mutual_info,
    p_value_adjusted_mutual_information_q_mc,
    p_value_adjusted_mutual_information_2_normal,
)
import scipy.sparse as sp
from scipy.special import erf

# Benchmarking imports
from random_partition import RandomSetPartition
from time import perf_counter
from tqdm import tqdm, trange
import signal
import sys

sys.setrecursionlimit(100_000_000)


class Timeout:
    def __init__(self, seconds=300, error_message="Timeout"):
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


def _get_kwargs(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    contingency: Optional[sp.csc_matrix],
    seed: Union[None, SeedSequence, BitGenerator, Generator],
    kwargs: Dict[str, Any],
    function: Callable,
) -> Dict[str, Any]:
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
    kwargs["labels_true"] = labels_true
    kwargs["labels_pred"] = labels_pred
    kwargs["contingency"] = contingency
    kwargs["seed"] = seed

    return {
        filter_key: kwargs[filter_key]
        for filter_key in signature_keys
        if filter_key in kwargs.keys()
    }


def _romano_run_trial(args):
    global set_partition_generator
    global comparison_metrics

    number_datapoints, reference_number_clusters, compare_number_clusters, seed = args

    # Generate reference clustering
    reference_labels = []
    for cluster in range(reference_number_clusters):
        reference_labels.extend(
            [cluster] * (number_datapoints // reference_number_clusters)
        )
    prng = default_rng(seed)
    set_partition_generator.reseed(prng)
    prng.shuffle(reference_labels)

    # Generate clusterings to compare to reference clustering
    comparison_labels = [
        set_partition_generator.random_partition(number_datapoints, number_clusters)
        for number_clusters in compare_number_clusters
    ]

    results = []
    for metric_name, metric, kwargs in comparison_metrics:
        # Compute metrics for all clusterings
        metric_values = [
            metric(
                **_get_kwargs(
                    reference_labels, comparison_label, None, prng, kwargs, metric
                )
            )
            for comparison_label in comparison_labels
        ]
        metric_values = [
            value[0] if isinstance(value, tuple) else value for value in metric_values
        ]
        selected_number_of_clusters = compare_number_clusters[np.argmax(metric_values)]
        results.append((metric_name, selected_number_of_clusters, metric_values))
    return results


def romano_experiment(
    comparison_metrics: List[Tuple[str, Callable, Dict[str, Any]]],
    number_datapoints: int = 500,
    reference_number_clusters: int = 10,
    compare_number_clusters: List[int] = [2, 6, 10, 14, 18, 22],
    trials: int = 5_000,
    seed: SeedSequence = SeedSequence(),
    n_jobs: Union[None, int] = None,
) -> pd.DataFrame:
    """Run the experiment performed in S. Romano et al. "Standardized Mutual Information
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
        raise ValueError("Number of datapoints must be divisible by number of clusters")

    if n_jobs is None:
        n_jobs = cpu_count()

    params = (
        (
            number_datapoints,
            reference_number_clusters,
            compare_number_clusters,
            seed.spawn(1)[0],
        )
        for _ in range(trials)
    )

    with Pool(
        n_jobs, initializer=_worker_init_fn, initargs=(comparison_metrics,)
    ) as pool:
        results = list(
            tqdm(
                chain.from_iterable(
                    pool.imap_unordered(
                        _romano_run_trial,
                        params,
                        chunksize=max(1, trials // (100 * n_jobs)),
                    )
                ),
                total=trials * len(comparison_metrics),
            )
        )

    df = pd.DataFrame(
        results, columns=["metric_name", "number_clusters", "metric_values"]
    )
    return df


def standardization_vs_p_value(
    n_samples: int,
    n_elements: List[int],
    accuracy_goal: float = 1e-3,
    seed=SeedSequence(),
) -> pd.DataFrame:
    """Experiment to compare the standardization approach to the p-value approach.

    We generate random clusterings and compare the Monte Carlo and normal approximation
    for the PMI2 metric.

    Args:
        n_samples: Number of samples to generate
        n_elements: List of number of elements in the clusterings
        seed: random seed sequence to use for reproducibility

    Returns:
        Dataframe with raw results
    """
    prng = default_rng(seed=seed)

    clustering_dist = RandomSetPartition(prng)

    results = {"PMI2_normal": [], "PMI2_mc": [], "PMI2_mc_err": [], "n": []}

    for n in n_elements:
        print(f"n={n}")
        for _ in trange(n_samples):
            labels_true = clustering_dist.random_partition(n)
            labels_pred = clustering_dist.random_partition(n)
            sri = p_value_adjusted_mutual_information_2_normal(labels_true, labels_pred)
            p_value, p_err = p_value_adjusted_mutual_information_q_mc(
                labels_true, labels_pred, q=2.0, seed=prng, accuracy_goal=accuracy_goal
            )
            results["PMI2_normal"].append(sri)
            results["PMI2_mc"].append(p_value)
            results["PMI2_mc_err"].append(p_err)
            results["n"].append(n)

    df = pd.DataFrame(results)
    return df


def _synthetic_rcn_run_trial(
    args: Tuple[int, int, int, SeedSequence]
) -> List[Tuple[int, int, int, str, float, float]]:
    global set_partition_generator
    global comparison_metrics

    r, c, n, seed = args

    prng = default_rng(seed)
    # Global partition generator for caching partition function calls.
    set_partition_generator.reseed(prng)

    labels_true = np.array(set_partition_generator.random_partition(n, r))
    labels_pred = np.array(set_partition_generator.random_partition(n, c))

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    m = contingency.count_nonzero()

    results = []

    for metric_name, metric, kwargs in comparison_metrics:
        kwargs = _get_kwargs(
            labels_true, labels_pred, contingency, prng, kwargs, metric
        )
        value = np.nan
        value_error = np.nan
        timing = np.nan
        try:
            with Timeout(seconds=1200):
                start_time = perf_counter()
                value = metric(**kwargs)
                end_time = perf_counter()
                timing = end_time - start_time
        except Exception:
            pass
        if isinstance(value, tuple):
            value, value_error = value
        results.append((r, c, n, m, metric_name, value, value_error, timing))

    return results


def synthetic_rcn_benchmark(
    comparison_metrics: List[Tuple[str, Callable]],
    rcn_grid: List[Tuple[Optional[int], Optional[int], int]],
    trials: int = 20,
    seed: SeedSequence = SeedSequence(),
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Run a benchmark on synthetic random data with fixed R,C,N.

    The benchmark is repeated 20 times for each R,C,N combination with two random
    clusterings generated for each trial.

    Args:
        comparison_metrics: List of tuples of metric name and metric function
        rcn_grid: List of tuples of R, C, N to benchmark on.
        trials: Number of trials per R, C, N combination
        seed: random seed sequence to use for reproducibility
        n_jobs: Number of jobs to run in parallel (None for all available)

    Returns:
        Dataframe with results + timings.
    """
    parameters = (
        (r, c, n, seed.spawn(1)[0]) for r, c, n in rcn_grid for _ in range(trials)
    )
    total_iterations = len(rcn_grid) * trials

    if n_jobs is None:
        n_jobs = cpu_count()

    with Pool(
        n_jobs, initializer=_worker_init_fn, initargs=(comparison_metrics,)
    ) as pool:
        results = list(
            tqdm(
                chain.from_iterable(
                    pool.imap_unordered(
                        _synthetic_rcn_run_trial, parameters, chunksize=1
                    )
                ),
                total=total_iterations * len(comparison_metrics),
            )
        )

    return pd.DataFrame(
        results,
        columns=[
            "R",
            "C",
            "N",
            "m",
            "metric_name",
            "metric_value",
            "metric_error",
            "timing",
        ],
    )


if __name__ == "__main__":
    n_jobs = 16
    seed = SeedSequence(42)

    rcn_seed, romano_seed, n_seed = seed.spawn(3)

    # Romano Experiment
    metrics = [
        ("RI", rand_score, dict()),
        ("ARI", adjusted_rand_score, dict()),
        (
            "PMI2",
            p_value_adjusted_mutual_information_q_mc,
            {"q": 2, "accuracy_goal": 1e-3},
        ),
        ("MI", mutual_info_score, dict()),
        ("AMI", adjusted_mutual_info_score, dict()),
        (
            "PMI1",
            p_value_adjusted_mutual_information_q_mc,
            {"q": 1, "accuracy_goal": 1e-3},
        ),
    ]

    print("Romano experiment ...")
    df = romano_experiment(
        metrics,
        number_datapoints=500,
        reference_number_clusters=10,
        compare_number_clusters=[2, 6, 10, 14, 18, 22],
        trials=5_000,
        seed=romano_seed,
    )
    df.to_csv("results/romano.csv", index=False)
    del df

    # Standardization vs p-value experiment
    print("Standardization vs p-value experiment ...")
    df = standardization_vs_p_value(
        n_samples=1000,
        n_elements=[50, 100, 200, 500, 1000],
        accuracy_goal=1e-3,
        seed=42,
    )
    df.to_csv("results/standardization_vs_p_value.csv", index=False)
    del df

    # Runtime Experiment 1
    metrics = [
        ("SMI1", standardized_mutual_info, dict()),
        ("SMI2", standardized_rand_score, dict()),
        (
            "PMI2",
            p_value_adjusted_mutual_information_q_mc,
            {"q": 2, "accuracy_goal": 1e-3},
        ),
    ]

    rcn_grid = [(10, 10, n) for n in np.logspace(1, 6, num=21, dtype=np.int64)]

    print("Running RCN benchmark with constant R,C and varying N ...")
    df = synthetic_rcn_benchmark(
        metrics, rcn_grid, seed=rcn_seed, trials=16, n_jobs=n_jobs
    )
    df.to_csv("results/rcn_benchmark.csv", index=False)
    del df

    # Runtime Experiment 2
    metrics = [
        (
            f"PMI2(a={a:.6f})",
            p_value_adjusted_mutual_information_q_mc,
            {"q": 2, "accuracy_goal": a},
        )
        for a in np.logspace(-4, 0, 21)
    ]
    metrics.append(("SMI2", standardized_rand_score, dict()))

    rcn_grid = [(None, None, 1000)]

    print("Running N benchmark with various accuracies ...")
    df = synthetic_rcn_benchmark(
        metrics, rcn_grid, seed=n_seed, trials=100, n_jobs=n_jobs
    )
    df.to_csv("results/n_benchmark_accuracy.csv", index=False)
    del df
