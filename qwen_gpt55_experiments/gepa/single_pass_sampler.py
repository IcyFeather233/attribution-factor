"""Use GEPA's sampler interface for one shuffled pass without padding."""
import random

from gepa.strategies.batch_sampler import EpochShuffledBatchSampler


class SinglePassBatchSampler(EpochShuffledBatchSampler):
    """One proposal per iteration; the final batch contains only remaining IDs.

    Subclassing the native sampler also preserves its fields through the existing
    engine checkpoint hook. This sampler intentionally stops after one pass.
    """

    def __init__(self, minibatch_size, seed=42):
        if minibatch_size < 1:
            raise ValueError('minibatch_size must be positive')
        super().__init__(minibatch_size, rng=random.Random(seed))

    def next_minibatch_ids(self, loader, state):
        if not self.shuffled_ids:
            self.shuffled_ids = list(loader.all_ids())
            self.last_trainset_size = len(loader)
            if not self.shuffled_ids:
                raise ValueError('Cannot sample an empty training set')
            self.rng.shuffle(self.shuffled_ids)
        if len(loader) != self.last_trainset_size:
            raise ValueError('Training set changed during single-pass optimization')
        if state.i == self._current_iteration:
            raise ValueError('Single-pass sampler requires one proposal per iteration')
        start = state.i * self.minibatch_size
        if state.i < 0 or start >= len(self.shuffled_ids):
            raise ValueError('Single-pass optimization exhausted the training set')
        self._current_iteration = state.i
        return self.shuffled_ids[start:start + self.minibatch_size]
