import numpy as np
from numpy.typing import ArrayLike, NDArray
from numpy.random import default_rng, SeedSequence, Generator, BitGenerator
from sklearn.metrics.cluster import contingency_matrix
from typing import Optional, Union, Tuple
from scipy import sparse as sp
from scipy.special import erf
from scipy.stats import random_table
from standardized_mutual_info import standardized_mutual_info_cython


def _tsallis_entropy(p: NDArray, q: float = 1.0, axis: Union[int, Tuple, None] = None):
    """Tsallis entropy of a probability distribution.

    Args:
        p (array):
            Probability distribution.
        q (float):
            The non-additivity q of the Tsallis entropy (1 for Shannon entropy, 2 for Rand Index).
        axis (int):
            The axis along which to compute the entropy.

    Returns:
        The Tsallis entropy.
    """
    if q == 1.0:
        return -np.sum(p * np.log(p, where=p > 0), axis=axis)
    return 1 / (q - 1) * (1 - (p**q).sum(axis=axis))


def p_value_adjusted_mutual_information_q_mc(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    q: float = 1.0,
    seed: Union[None, SeedSequence, BitGenerator, Generator] = None,
    accuracy_goal: float = 0.01,
    contingency: Optional[sp.csr_matrix] = None,
) -> Tuple[float, float]:
    """Monte Carlo estimate of the p-value of the generalized mutual information under random permutation of the labels.

    Args:
        labels_true (int array[n_samples]):
            A clustering of the data into disjoint subsets.
        labels_pred (int array[n_samples]):
            Another clustering of the data into disjoint subsets.
        q (float):
            The non-additivity q of the Tsallis entropy (1 for mutual information, 2 for Rand Index).
        seed (optional):
            Random seed.
        accuracy_goal (float):
            The desired accuracy of the Monte Carlo estimate.
        contingency (optional):
            The contingency matrix of the two clusterings.

    Returns:
        The p-value and the error of the Monte Carlo estimate.
    """
    prng = default_rng(seed=seed)

    if contingency is None:
        contingency: sp.csr_matrix = contingency_matrix(
            labels_true, labels_pred, sparse=True
        )

    a = np.ravel(contingency.sum(axis=1))
    b = np.ravel(contingency.sum(axis=0))
    n = contingency.sum()

    if len(a) == 1 or len(b) == 1:
        return np.nan, np.nan

    joint_entropy = _tsallis_entropy(contingency.data / n, q=q)
    batch_size = max(1, 200_000 // (len(a) * len(b)))

    contingency_dist = random_table(a, b)
    true_count = 0
    total_count = 0
    p_value_error = 1.0

    while total_count < 2_000 or p_value_error > accuracy_goal:
        contingencies = contingency_dist.rvs(size=batch_size, random_state=prng)

        joint_entropy_sample = _tsallis_entropy(contingencies / n, q=q, axis=(1, 2))
        true_count += np.sum(
            (joint_entropy_sample > joint_entropy)
            + 0.5 * (joint_entropy_sample == joint_entropy)
        )
        total_count += batch_size

        p_value = true_count / total_count
        # https://stats.stackexchange.com/questions/11541/how-to-calculate-se-for-a-binary-measure-given-sample-size-n-and-known-populati
        p_value_error = np.sqrt(p_value * (1 - p_value) / total_count)

    return p_value, p_value_error


def standardized_rand_score(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    contingency: Optional[sp.csr_matrix] = None,
) -> float:
    """Standardized Rand index for two clusterings under the random permutation model.

    Args:
        labels_true (int array[n_samples]):
            A clustering of the data into disjoint subsets.
        labels_pred (int array[n_samples]):
            Another clustering of the data into disjoint subsets.
        contingency (optional):
            Sparse contingency matrix.

    Returns:
        The standardized Rand index.
    """
    n = len(labels_true)

    if n < 4:
        if n < 2:
            raise ValueError("Standardized Rand index is not defined for n_samples < 2")
        raise NotImplementedError(
            "Standardized Rand index is not implemented for n_samples < 4"
        )

    # Computation using the contingency data
    if contingency is None:
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    a = np.ravel(contingency.sum(axis=0))
    b = np.ravel(contingency.sum(axis=1))

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Standardized Rand index is not defined for n_clusters < 2")

    x = (contingency.data * (contingency.data - 1)).sum() / 2

    if max(a) > n - 2:
        a, b = b, a

    a_sum = (a * (a - 1)).sum()
    if max(b) > n - 2:
        ex = 0.5 * ((n - 2) / n) * a_sum
        ex2 = 0.25 * (a * (a_sum - 2 * (a - 1)) ** 2 / n).sum()
    else:
        a2_sum = ((a * (a - 1)) ** 2).sum()
        b_sum = (b * (b - 1)).sum()
        b2_sum = ((b * (b - 1)) ** 2).sum()
        ex = a_sum * b_sum / (2 * n * (n - 1))
        normalizer = (n - 1) * (n - 2) * (n - 3)

        ij = (
            2 * a_sum * ((n - b) * (n - 3 * (b - 1)) * (b - 1) * b / normalizer).sum()
            + (a**2 * (a - 1)).sum()
            * ((4 * n - 5 * b + 3) * (b - 2) * (b - 1) * b / normalizer).sum()
            + (a**3 * (a - 1) / normalizer).sum()
            * ((b - 3) * (b - 2) * (b - 1) * b).sum()
        )
        ipj = (b * (b - 1) * (b - 2) * (b - 3) / normalizer).sum() * (
            a_sum**2 - a2_sum
        )
        ijp = (a * (a - 1) * (a - 2) * (a - 3) / normalizer).sum() * (
            b_sum**2 - b2_sum
        )
        ipjp = (a_sum**2 - a2_sum) / normalizer * (b_sum**2 - b2_sum)
        ex2 = (ij + ipj + ijp + ipjp) / (4 * n)

    var_x = ex2 - ex**2
    # Analytical continuation for zero variance (accounting for numerical errors)
    if var_x < 1e-10:
        # If there is no variance, the expected value is the observed value
        # such that the analytical continuation is 1.0
        return 1.0

    return (x - ex) / np.sqrt(var_x)


def standardized_mutual_info(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    contingency: Optional[sp.csr_matrix] = None,
) -> float:
    """Standardized mutual information for two clusterings under pairwise permutations.

    This code is based on the paper "Standardized Mutual Information for Clustering
    Comparisons: One Step Further in Adjustment for Chance" by S. Romano et al.
    (https://proceedings.mlr.press/v32/romano14.html).

    Args:
        labels_true (int array[n_samples]):
            A clustering of the data into disjoint subsets.
        labels_pred (int array[n_samples]):
            Another clustering of the data into disjoint subsets.
        contingency (optional):
            Sparse contingency matrix.

    Returns:
        The standardized mutual information.
    """
    n = len(labels_true)
    if contingency is None:
        contingency = contingency_matrix(
            labels_true, labels_pred, sparse=True, dtype=np.int64
        )
    return standardized_mutual_info_cython(contingency, n)


def p_value_adjusted_mutual_information_2_normal(
    labels_true: ArrayLike,
    labels_pred: ArrayLike,
    contingency: Optional[sp.csr_matrix] = None,
) -> float:
    """Normal estimate of the p-value of the rand index under random permutation of the labels.

    Args:
        labels_true (int array[n_samples]):
            A clustering of the data into disjoint subsets.
        labels_pred (int array[n_samples]):
            Another clustering of the data into disjoint subsets.
        contingency (optional):
            The contingency matrix of the two clusterings.

    Returns:
        The p-value estimate.
    """
    return 0.5 * (
        1
        + erf(
            standardized_rand_score(labels_true, labels_pred, contingency) / np.sqrt(2)
        )
    )
