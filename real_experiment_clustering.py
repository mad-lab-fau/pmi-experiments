import argparse

from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.cluster import (
    KMeans,
    SpectralClustering,
    AgglomerativeClustering,
    BisectingKMeans,
)
from sklearn.datasets import (
    fetch_olivetti_faces,
    fetch_openml,
    load_digits,
)
from sklearn.metrics.cluster import rand_score, adjusted_rand_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tqdm import tqdm
from clustering_comparison_measures import standardized_rand_score
import warnings

warnings.filterwarnings("ignore")

metric_names = ["ri", "ari", "sri"]


def get_dataset(name: str):
    if name == "olivetti":
        return fetch_olivetti_faces(data_home="data", return_X_y=True)
    if name == "digits":
        return load_digits(return_X_y=True)

    openml_ids = {
        "segment": 40984,
        "texture": 40499,
    }

    if name in openml_ids:
        X, y = fetch_openml(
            data_id=openml_ids[name],
            data_home="data",
            return_X_y=True,
            parser="auto",
            as_frame=True,
        )
        # Convert every categorical feature to integer
        for column in X.columns:
            if X[column].dtype == "category":
                X[column] = X[column].cat.codes
        # Convert categorical label to integer
        y = y.astype("category").cat.codes
        return X.to_numpy(), y.to_numpy()


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
        action="store_true",
        help="Whether script should operate in download mode",
    )
    parser.add_argument(
        "--subsampling",
        "-s",
        action="store_true",
        help="Whether or not to use subsets of the datasets",
    )
    parser.add_argument(
        "--algorithms",
        "-a",
        nargs="+",
        type=str,
        default="kmeans",
        help="Clustering algorithms to use [kmeans, spectral, agglomerative, ward, bisectingkmeans]",
    )
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        type=str,
        default=["olivetti", "digits"],
        help="Datasets to use [olivetti, digits, segment, texture]",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default="clustering.csv",
        help="Output file for benchmark results",
    )
    args = parser.parse_args()
    if isinstance(args.algorithms, str):
        args.algorithms = [args.algorithms]
    if isinstance(args.algorithms, str):
        args.datasets = [args.datasets]

    need_subsampling = {"agglomerative", "ward"}
    if (not args.subsampling) and (set(args.algorithms) & need_subsampling):
        print(
            f"Algorithms {set(args.algorithms) & need_subsampling} require subsampling."
        )
        exit(1)

    # Run benchmark
    records = []  # list of result tuples
    for name in args.datasets:
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

        algorithms = deepcopy(args.algorithms)
        if (
            ("spectral" in algorithms)
            and (num_samples > 1_000)
            and (not args.subsampling)
        ):
            print(f"Skipping spectral clustering for {name} (too many samples)")
            algorithms.remove("spectral")

        for i in tqdm(range(args.repetitions)):
            if args.subsampling:
                subsamples = min(1_000, 0.8 * num_samples)
                data_resampled, labels_resampled = resample(
                    data,
                    labels_true,
                    n_samples=subsamples,
                    random_state=i,
                    replace=False,
                    stratify=labels_true,
                )
                num_clusters = np.unique(labels_resampled).size
            else:
                data_resampled = data
                labels_resampled = labels_true
                subsamples = num_samples
            for k in np.unique(
                np.linspace(0.5 * num_clusters, 1.5 * num_clusters, 11).round()
            ).astype(int):
                for algorithm_name in algorithms:
                    if algorithm_name == "kmeans":
                        algorithm = KMeans(n_clusters=k, n_init="auto", random_state=i)
                    elif algorithm_name == "spectral":
                        algorithm = SpectralClustering(
                            n_clusters=k,
                            eigen_solver="amg",
                            random_state=i,
                            affinity="nearest_neighbors",
                            assign_labels="cluster_qr",
                            n_jobs=-1,
                        )
                    elif algorithm_name == "agglomerative":
                        algorithm = AgglomerativeClustering(
                            n_clusters=k, linkage="single"
                        )
                    elif algorithm_name == "ward":
                        algorithm = AgglomerativeClustering(
                            n_clusters=k, linkage="ward"
                        )
                    elif algorithm_name == "bisectingkmeans":
                        algorithm = BisectingKMeans(n_clusters=k, random_state=i)

                    estimator = make_pipeline(StandardScaler(), algorithm).fit(
                        data_resampled
                    )
                    labels_pred = estimator[-1].labels_

                    # Compute different metrics
                    metric_vals = [
                        get_metric(name, labels_resampled, labels_pred)
                        for name in metric_names
                    ]
                    records.append(
                        (name, algorithm_name, i, num_clusters, k, *metric_vals)
                    )
    # Save results as .csv
    columns = ["dataset", "algorithm", "rep_id", "k_true", "k_used"]
    columns.extend(metric_names)

    df = pd.DataFrame.from_records(records, columns=columns)
    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
