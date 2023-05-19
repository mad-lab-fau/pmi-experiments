from typing import Union, List, Optional
from numpy import ones, append
from numpy.typing import ArrayLike, NDArray
from numpy.random import default_rng, SeedSequence, BitGenerator, Generator
from scipy.optimize import root_scalar
import sys
from mpmath import stirling2, bell, factorial, power, inf, nsum, floor
from mpmath import exp as mpexp
sys.setrecursionlimit(100_000_000)

class RandomSetPartition:
    """Class to generate random partitions of a set."""

    def __init__(self, seed: Union[None, SeedSequence, BitGenerator, Generator] = None) -> None:
        self._prng = default_rng(seed)
        self._cache = {}

    def _count_partitions(self, n: int, k: int) -> int:
        """Return the number of set partitions of n into k parts.

        This function uses a cache to speed up the computation.

        Args:
            n: integer to be partitioned
            k: number of parts

        Returns:
            Number of partitions of n with k parts
        """
        if (n == k) or (k == 1):
            return 1
        if (n, k) in self._cache:
            return self._cache[n, k]
        if n > 5_000:
            # Limit Cache size to < 1 GB (guessing 8 bytes per int)
            return int(stirling2(n, k))
        x = self._count_partitions(n - 1, k - 1) + \
            k * self._count_partitions(n - 1, k)
        self._cache[n, k] = x
        return x

    def _bell_number(self, n: int) -> int:
        """Return the nth Bell number.

        The Bell number is the number of partitions of a set with n elements.

        Args:
            n: integer

        Returns:
            Bell number of n
        """
        if n not in self._cache:
            self._cache[n] = int(bell(n))
        return self._cache[n]

    def _random_partition_knuth(self, n: int) -> NDArray:
        """Return a random set partition of a set via Dobiński's formula.

        Args:
            n: size of the set to be partitioned

        Returns:
            An array of length n that represents the partition.
        """
        # Using Dobiński's formula
        # https://stats.stackexchange.com/questions/497858/sampling-uniformly-from-the-set-of-partitions-of-a-set
        u = self._prng.random()
        e_bell = self._bell_number(n) * mpexp(1)
        if u < (self._bell_number(n) - 1) / self._bell_number(n):
            # This will complete in O(n) time
            total = 0
            k = 1
            # Probably there is a better way than linear search, maybe using Stirlings Formula or something
            while total / e_bell < u:
                k += 1
                total += power(k, n) / factorial(k)
            k -= 1
        else:
            # Root finding if we are in the tail
            x = root_scalar(lambda x: (nsum(lambda k: power(
                k, n) / factorial(k), [1, floor(1 / x)]) / e_bell) - u if x != 0 else 1, bracket=[0, 1]).root
            k = int(floor(1 / x))
        # Now assign n items to k bins
        return self._prng.integers(0, k, size=n)

    def _random_partition_brute_force(self, n: int, k: int, min_label: int = 0) -> NDArray:
        """Return a random partition of an integer via brute force.

        Args:
            n: size of the set to be partitioned
            k: number of parts
            min_label: minimum label to be used

        Returns:
            An array of length n that represents the partition.
        """
        if k == 1:
            return ones(n, dtype=int) * min_label
        if self._prng.random() < self._count_partitions(n - 1, k - 1) / self._count_partitions(n, k):
            # n is a singleton in the partition
            return append(self._random_partition_brute_force(n - 1, k - 1, min_label + 1), min_label)
        else:
            # n is in a partition with more than one element
            partition = self._random_partition_brute_force(n - 1, k, min_label)
            return append(partition, self._prng.integers(min_label, k + min_label))

    def random_partition(self, n: int, k: Optional[int] = None) -> List[int]:
        """Return a random partition of a set of size n.

        Args:
            n: size of the set to be partitioned
            k: number of parts (None for an unconstrained partition)

        Returns:
            An array of length n that represents the partition.
        """
        if k is None:
            return self._random_partition_knuth(n)
        else:
            return self._random_partition_brute_force(n, k)

    def reseed(self, seed: Union[None, SeedSequence, BitGenerator, Generator] = None) -> None:
        """Reseed the random number generator.

        Args:
            seed: Seed for the random number generator. If None, a random seed is used.
        """
        self._prng = default_rng(seed)