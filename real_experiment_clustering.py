import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import (
    fetch_olivetti_faces,
    fetch_20newsgroups_vectorized,
    fetch_covtype,
    fetch_openml,
    load_digits,
)
from sklearn.metrics.cluster import rand_score, adjusted_rand_score, mutual_info_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from clustering_comparison_measures import standardized_rand_score

dataset_names = ["olivetti", "digits"]
metric_names = ["ri", "ari", "sri"]


def get_dataset(name: str):
    if name == "olivetti":
        return fetch_olivetti_faces(data_home="data", return_X_y=True)
    if name == "20news":
        return fetch_20newsgroups_vectorized(
            data_home="data", return_X_y=True, remove=("headers", "footers", "quotes")
        )
    if name == "covtype":
        return fetch_covtype(data_home="data", return_X_y=True)
    if name == "digits":
        return load_digits(return_X_y=True)
    if name == "mnist":
        return fetch_openml(
            name="mnist_784",
            data_home="data",
            return_X_y=True,
            parser="auto",
            as_frame=False,
        )


def get_metric(name: str, labels_true, labels_pred):
    metric_mapping = {
        "ri": rand_score,
        "ari": adjusted_rand_score,
        "sri": standardized_rand_score,
    }
    if name == "sri" and len(np.unique(labels_pred)) < 2:
        return np.nan  # SRI not defined for n_clusters < 2
    result = metric_mapping[name](labels_true, labels_pred)
    if type(result) is tuple:
        result = result[0]
    return result


def main():
    # Get command-line arguments for benchmark
    parser = argparse.ArgumentParser(
        description="Benchmark suite for clustering metrics."
    )
    parser.add_argument(
        "--repetitions",
        "-r",
        type=int,
        default=1,
        help="How often a clustering algorithm should be evaluated",
    )
    parser.add_argument(
        "--download",
        "-d",
        action="store_true",
        help="Whether script should operate in download mode",
    )
    args = parser.parse_args()
    # Run benchmark
    records = []  # list of result tuples
    for name in dataset_names:
        data, labels_true = get_dataset(name)
        labels_true = labels_true.astype(int)
        num_clusters = (
            np.unique(labels_true).size
            if labels_true.ndim == 1
            else labels_true.shape[1]
        )
        num_samples = labels_true.shape[0]
        if args.download:
            print(
                "Downloaded {} (n_clusters={}, n_samples={})".format(
                    name, num_clusters, num_samples
                )
            )
            continue  # only download, no computations
        print(
            "Running on {} (n_clusters={}, n_samples={})".format(
                name, num_clusters, num_samples
            )
        )
        for i in tqdm(range(args.repetitions)):
            for k in np.unique(
                np.linspace(0.5 * num_clusters, 1.5 * num_clusters, 11).round()
            ).astype(int):
                # Run k-means clustering
                kmeans = KMeans(n_clusters=k, n_init="auto", random_state=i)
                estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
                labels_pred = estimator[-1].labels_
                # Compute different metrics
                metric_vals = [
                    get_metric(name, labels_true, labels_pred) for name in metric_names
                ]
                records.append((name, i, num_clusters, k, *metric_vals))
    # Save results as .csv
    columns = [
        "dataset",
        "rep_id",
        "k_true",
        "k_used",
        "ri",
        "ari",
        "sri",
    ]
    df = pd.DataFrame.from_records(records, columns=columns)
    df.to_csv("results/clustering.csv", index=False)


if __name__ == "__main__":
    main()
