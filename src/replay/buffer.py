"""Dual-stream experience replay buffer.

Stream 1 (Domain): Per-domain buffers with reservoir sampling.
Stream 2 (General): Fixed buffer of general-knowledge examples.
"""
import random
from typing import Any

Example = dict[str, Any]


class DomainReplayBuffer:
    def __init__(self, max_per_domain: int = 200):
        self.max_per_domain = max_per_domain
        self._buffers: dict[int, list[Example]] = {}
        self._counts: dict[int, int] = {}

    def add_domain(self, domain_id: int, examples: list[Example], seed: int = 42):
        rng = random.Random(seed + domain_id)
        if domain_id not in self._buffers:
            self._buffers[domain_id] = []
            self._counts[domain_id] = 0
        buf = self._buffers[domain_id]
        for ex in examples:
            self._counts[domain_id] += 1
            n = self._counts[domain_id]
            if len(buf) < self.max_per_domain:
                buf.append(ex)
            else:
                j = rng.randint(0, n - 1)
                if j < self.max_per_domain:
                    buf[j] = ex

    def sample(self, domain_id: int, n: int, rng: random.Random | None = None) -> list[Example]:
        rng = rng or random.Random()
        buf = self._buffers.get(domain_id, [])
        if not buf:
            return []
        return rng.choices(buf, k=min(n, len(buf)))

    def sample_all(self, total_n: int, rng: random.Random | None = None) -> list[Example]:
        rng = rng or random.Random()
        all_domains = list(self._buffers.keys())
        if not all_domains:
            return []
        sizes = [len(self._buffers[d]) for d in all_domains]
        total_size = sum(sizes)
        if total_size == 0:
            return []
        # Proportional allocation per domain, then pad/trim to exactly total_n
        allocations = [max(1, int(total_n * s / total_size)) for s in sizes]
        samples = []
        for d, n_from_d in zip(all_domains, allocations):
            samples.extend(self.sample(d, n_from_d, rng))
        # Trim excess
        if len(samples) > total_n:
            samples = rng.sample(samples, total_n)
        # Pad shortage by sampling more from all domains uniformly
        while len(samples) < total_n:
            extra_domain = rng.choice(all_domains)
            samples.extend(self.sample(extra_domain, 1, rng))
        return samples[:total_n]

    def size(self, domain_id: int) -> int:
        return len(self._buffers.get(domain_id, []))

    @property
    def seen_domains(self) -> list[int]:
        return list(self._buffers.keys())


class GeneralReplayBuffer:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: list[Example] = []

    def fill(self, examples: list[Example], seed: int = 42):
        rng = random.Random(seed)
        if len(examples) <= self.max_size:
            self._buffer = list(examples)
        else:
            self._buffer = rng.sample(examples, self.max_size)

    def sample(self, n: int, rng: random.Random | None = None) -> list[Example]:
        rng = rng or random.Random()
        if not self._buffer:
            return []
        return rng.choices(self._buffer, k=min(n, len(self._buffer)))

    def size(self) -> int:
        return len(self._buffer)


class DualReplayBuffer:
    def __init__(self, max_per_domain: int = 200, general_max_size: int = 1000):
        self.domain_buffer = DomainReplayBuffer(max_per_domain)
        self.general_buffer = GeneralReplayBuffer(general_max_size)

    def add_domain(self, domain_id: int, examples: list[Example], seed: int = 42):
        self.domain_buffer.add_domain(domain_id, examples, seed)

    def fill_general(self, examples: list[Example], seed: int = 42):
        self.general_buffer.fill(examples, seed)

    def sample_replay(self, domain_n: int, general_n: int, rng: random.Random | None = None) -> tuple[list[Example], list[Example]]:
        rng = rng or random.Random()
        domain_samples = self.domain_buffer.sample_all(domain_n, rng)
        general_samples = self.general_buffer.sample(general_n, rng)
        return domain_samples, general_samples
