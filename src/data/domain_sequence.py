"""Domain ordering generation for sequential training."""
import random


def generate_domain_orderings(
    num_domains: int, num_orderings: int, seed: int = 42
) -> list[list[int]]:
    """Generate reproducible random domain orderings.

    Each ordering is a permutation of [0, num_domains-1].
    Different orderings use different sub-seeds (seed+i) to guarantee
    they are distinct while remaining fully reproducible.
    """
    orderings = []
    for i in range(num_orderings):
        rng = random.Random(seed + i)
        order = list(range(num_domains))
        rng.shuffle(order)
        orderings.append(order)
    return orderings
