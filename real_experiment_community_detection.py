import argparse
import os
import shutil
import gzip
import requests
import csv
from contextlib import redirect_stdout, redirect_stderr
from itertools import product
from pathlib import Path
from typing import Dict

import networkit as nk
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import rand_score, adjusted_rand_score
from tqdm import tqdm

from clustering_comparison_measures import standardized_rand_score


class EmailCoreDataset:
    def __init__(self) -> None:
        self._base_dir = Path("data")
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._graph_url = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
        self._labels_url = (
            "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"
        )
        self._graph = None
        self._node_map = None

    def _download(self, url: str, typename: str) -> None:
        gz_file = self._base_dir / f"{self.name}_{typename}.txt.gz"
        txt_file = self._base_dir / f"{self.name}_{typename}.txt"

        with requests.get(url, stream=True) as r:
            total_size_in_bytes = int(r.headers.get("content-length", 0))
            r.raise_for_status()
            with open(gz_file, "wb") as f:
                for chunk in tqdm(
                    r.iter_content(chunk_size=8192),
                    total=total_size_in_bytes / 8192,
                    desc=f"Downloading {self.name}_{typename}",
                    unit_scale=True,
                ):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)

        print("Unzipping ...")
        # Popen(
        #     f"gunzip {gz_file.resolve()}", shell=True).wait()
        with gzip.open(gz_file, "rb") as f_in:
            with open(txt_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    @classmethod
    @property
    def name(cls) -> str:
        "email"

    @property
    def graph(self) -> nk.Graph:
        if self._graph is None:
            path = self._base_dir / f"{self.name}_ungraph.txt"
            if not path.is_file():
                self._download(self._graph_url, typename="ungraph")
            reader = nk.graphio.EdgeListReader(
                "\t", 0, commentPrefix="#", continuous=False, directed=False
            )
            self._graph = reader.read(str(path.resolve()))
            self._node_map = reader.getNodeMap()
        return self._graph

    @property
    def node_map(self) -> Dict[str, int]:
        if self._node_map is None:
            self.graph
        return self._node_map

    @property
    def labels(self) -> nk.structures.Partition:
        path = self._base_dir / f"{self.name}_cmty.txt"
        if not path.is_file():
            self._download(self._labels_url, "cmty")
        return nk.graphio.EdgeListPartitionReader(firstNode=0, sepChar=" ").read(
            str(path.resolve())
        )


def export_graph(graph: nk.Graph, labels: nk.Cover | nk.Partition, name: str):
    """Saves graph as GML with community labels as node properties."""
    nx_graph = nk.nxadapter.nk2nx(graph)
    attrs = {n: labels.subsetOf(n) for n in graph.iterNodes()}
    nx.set_node_attributes(nx_graph, attrs, "community")
    nx.write_gml(nx_graph, "{}.gml".format(name))


def create_subgraph(
    graph: nk.Graph, labels: nk.Partition, cgraph: nk.Graph, num_communities=10
):
    """Creates a random connected subgraph with the given amount of communities."""
    collected_nodes, collected_labels, collected_communities = [], [], []
    community = np.random.randint(0, cgraph.numberOfNodes())
    while len(collected_communities) < num_communities:
        collected_communities.append(community)
        new_nodes = labels.getMembers(community)
        collected_nodes.extend(new_nodes)
        collected_labels.extend([community] * len(new_nodes))
        # Find next candidate node (connected to current graph + non-visited community)
        neighbors = set()
        for c in collected_communities:
            neighbors.update(cgraph.iterNeighbors(c))
        unvisited = [n for n in neighbors if n not in collected_communities]
        if len(unvisited) == 0:
            return None  # number of connected communities too small
        else:
            community = np.random.choice(unvisited)
    # Create subgraph with labels
    num_nodes = len(collected_nodes)
    sub_graph = nk.graphtools.subgraphFromNodes(graph, collected_nodes, compact=True)
    sub_labels = nk.Partition(num_nodes)
    for i in range(num_nodes):
        sub_labels.addToSubset(collected_labels[i], i)
    return sub_graph, sub_labels


def create_subgraph_retry(
    graph: nk.Graph, labels: nk.Partition, cgraph: nk.Graph, num_communities=10
):
    """Tries to create a suitable subgraph and repeats if it fails initially."""
    subgraph = None
    while subgraph is None:
        subgraph = create_subgraph(graph, labels, cgraph, num_communities)
    return subgraph


def main():
    # Get command-line arguments for benchmark
    parser = argparse.ArgumentParser(
        description="Benchmark for clustering comparison measures."
    )
    parser.add_argument(
        "--communities",
        "-c",
        type=int,
        default=30,
        help="Number of communities in each sub-sampled graph",
    )
    parser.add_argument(
        "--repetitions",
        "-r",
        type=int,
        default=100,
        help="How often a sub-sampled graph should be created and evaluated",
    )

    args = parser.parse_args()
    # Benchmark
    records = []  # list of result tuples
    dataset = EmailCoreDataset()
    graph, labels = dataset.graph, dataset.labels
    labels.compact()  # Required to prevent subsequent segmentation faults
    community_graph = nk.community.communityGraph(graph, labels)
    exp_name = "email_{}".format(args.communities)
    os.makedirs(exp_name, exist_ok=True)
    # export_graph(graph, labels, 'email/full')  # Full graph with community labels
    for i in tqdm(range(args.repetitions), disable=True):
        # Create a subgraph with the required number of communities
        print("Creating subgraph {}".format(i), flush=True)
        sub_graph, sub_labels = create_subgraph_retry(
            graph, labels, community_graph, args.communities
        )
        print("Created subgraph {}".format(i), flush=True)
        export_graph(
            sub_graph, sub_labels, "{}/sub_{}".format(exp_name, i)
        )  # Subgraph with community labels
        # Extract ground-truth labels for subgraph
        labels_true = [sub_labels.subsetOf(n) for n in sub_graph.iterNodes()]
        # Use desired algorithm to estimate communities
        algorithms = {
            "plm": nk.community.PLM,
            "plp": nk.community.PLP,
            "map": nk.community.LouvainMapEquation,
            "lpd": nk.community.LPDegreeOrdered,
            "pld": nk.community.ParallelLeiden,
        }
        parameters = {
            "plm": {
                "gamma": np.logspace(-3, 0, 4).tolist(),
            },
            "plp": {},
            "map": {
                "hierarchical": [True, False],
            },
            "lpd": {},
            "pld": {
                "randomize": [True, False],
                "gamma": np.logspace(-6, 0, 7).tolist(),
            },
        }
        for algo_name, algo in algorithms.items():
            algo_params = parameters[algo_name]
            for params in product(*algo_params.values()):
                params = dict(zip(algo_params.keys(), params))
                base_params = params.copy()
                # Perform community detection
                print(
                    "Running community detection {} ({})".format(
                        algo_name, base_params
                    ),
                    flush=True,
                )
                with redirect_stdout(None):
                    with redirect_stderr(None):
                        communities = nk.community.detectCommunities(
                            graph, algo(graph, **params), inspect=False
                        )
                print(
                    "Ran community detection {} ({})".format(algo_name, base_params),
                    flush=True,
                )
                # Extract predicted labels for subgraph
                labels_pred = [communities.subsetOf(n) for n in sub_graph.iterNodes()]
                # Compute different metrics
                print("Computing metrics", flush=True)
                ri = rand_score(labels_true, labels_pred)
                ari = adjusted_rand_score(labels_true, labels_pred)
                sri = standardized_rand_score(labels_true, labels_pred)

                print("Computed metrics", flush=True)
                records.append(
                    (
                        i,
                        sub_graph.numberOfNodes(),
                        args.communities,
                        communities.numberOfSubsets(),
                        algo_name,
                        base_params,
                        ri,
                        ari,
                        sri,
                    )
                )
    # Save results as .csv
    columns = [
        "rep_id",
        "n_nodes",
        "k_true",
        "k_pred",
        "algo",
        "params",
        "ri",
        "ari",
        "sri",
    ]
    df = pd.DataFrame.from_records(records, columns=columns)
    df.to_csv("community_detection_{}.csv".format(exp_name), index=False)


if __name__ == "__main__":
    main()
