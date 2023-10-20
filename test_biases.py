from other_indices import Indices
from random_partition import RandomSetPartition
from mpmath import stirling2
from clustering_comparison_measures import (
    standardized_rand_score,
    standardized_mutual_info,
)
from other_indices.ConstantBaselineTests import check_constant_baseline
from other_indices.ValidationIndices import Score
from other_indices.Clustering import Clustering
from fractions import Fraction
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class StandardizedMutualInformation(Score):
    @classmethod
    def score(cls, A, B):
        A, B = (Clustering.FromAnything(C) for C in [A, B])
        return standardized_mutual_info(A, B)


class StandardizedRandIndex(Score):
    @classmethod
    def score(cls, A, B):
        A, B = (Clustering.FromAnything(C) for C in [A, B])
        return standardized_rand_score(A, B)


def generate_k_set_partitions(ns, k):
    """Generate all k-set partitions of the set ns.

    This algorithm is described in Donald Knuth, Computer Programming,
    Volume 4, Fascicle 3B and the implementation is adapted from
    https://codereview.stackexchange.com/questions/1526/finding-all-k-subset-partitions.

    Args:
        ns: list of integers
        k: number of parts
    """

    def visit(n, a):
        ps = [[] for i in range(k)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, k + 1):
        a[n - k + j] = j - 1
    return f(k, n, 0, n, a)


AllIndices = Indices + [StandardizedMutualInformation, StandardizedRandIndex]

results = {index.__name__: 0 for index in AllIndices}

if __name__ == "__main__":
    tqdm.write("Type I biased:")
    p = tqdm(AllIndices, total=len(AllIndices), desc="Testing Type I bias")
    for index in p:
        p.set_description(f"Testing {index.__name__} for Type I bias")
        # Choose n=20,30,40
        ns = range(20, 51, 10)
        # For each n, we consider balanced cluster sizes with k=sqrt(n) clusters.
        n2gtk = {n: int(n**0.5) for n in ns}
        # For each n, we consider candidates with balanced cluster sizes with
        # k1=n^0.25, k2=n^0.5, k3=n^0.75.
        n2ks = {n: [int(n**0.25), int(n**0.5), int(n**0.75)] for n in ns}
        result = check_constant_baseline(
            I=index, n2ks=n2ks, n2gtk=n2gtk, repeats=500, aggregate=True
        )
        if result["constant baseline p"] < 1e-6:
            tqdm.write(f" - {index.__name__}")
    p.close()

    tqdm.write("Checking Type II bias with the following parameters:")
    # Type II bias
    n = 4
    k1 = 2
    k2 = 3

    random_partition = RandomSetPartition(seed=42)
    a = [int(i) for i in random_partition.random_partition(n, k=2)]

    tqdm.write(f"n={n}, k1={k1}, k2={k2}")
    tqdm.write(f"A = {a}")

    for b in tqdm(
        generate_k_set_partitions(list(range(n)), k1),
        total=int(stirling2(n, k1)),
        desc="Testing Type II bias",
    ):
        for b_prime in generate_k_set_partitions(list(range(n)), k2):
            for index in AllIndices:
                val = index.score(a, b)
                val_prime = index.score(a, b_prime)
                results[index.__name__] += int(val > val_prime) - int(val < val_prime)

    total = int(stirling2(n, k1) * stirling2(n, k2))

    tqdm.write("\nType II biased:")
    for key, value in results.items():
        frac = Fraction(results[key] + total, 2 * total)
        numerator, denominator = frac.as_integer_ratio()
        if (numerator, denominator) != (1, 2):
            tqdm.write(f"- {key}\tE = {numerator} / {denominator}")
