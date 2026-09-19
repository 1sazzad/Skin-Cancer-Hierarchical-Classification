"""Preserve persistent worker RNG streams across complete-epoch restarts."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info


class _Dataset(Dataset):
    def __init__(self, dataset, states):
        self.dataset = dataset
        self.states = states

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class _WorkerInit:
    def __init__(self, original):
        self.original = original

    def __call__(self, worker_id):
        if self.original is not None:
            self.original(worker_id)
        state = get_worker_info().dataset.states.get(worker_id)
        if state is not None:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"].cpu())


class _Collate:
    def __init__(self, original):
        self.original = original

    def __call__(self, samples):
        batch = self.original(samples)
        worker = get_worker_info()
        return batch, (None if worker is None else worker.id), {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }


class ResumableTrainLoader:
    """Keep the original sampler and batch policy; transport worker RNG metadata."""

    def __init__(self, original, worker_states=None, *, resumed=False):
        self.generator = original.generator
        self.worker_states = dict(worker_states or {}) if original.persistent_workers else {}
        self.dataset = original.dataset
        # Separate worker base-seed draws from the sampler on subsequent process
        # starts. The first epoch consumes exactly the original base-seed draw.
        worker_generator = torch.Generator()
        worker_generator.set_state(self.generator.get_state())
        if not resumed or not original.persistent_workers:
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        self.loader = DataLoader(
            _Dataset(original.dataset, self.worker_states if original.persistent_workers else {}),
            batch_sampler=original.batch_sampler,
            num_workers=original.num_workers,
            collate_fn=_Collate(original.collate_fn),
            pin_memory=original.pin_memory,
            timeout=original.timeout,
            worker_init_fn=_WorkerInit(original.worker_init_fn),
            multiprocessing_context=original.multiprocessing_context,
            generator=worker_generator,
            prefetch_factor=original.prefetch_factor,
            persistent_workers=original.persistent_workers,
        )

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        # Nonpersistent loaders draw a new base seed each epoch. Mirror that
        # consumption in the sampling generator after the initial construction.
        if not self.loader.persistent_workers and getattr(self, "iterated", False):
            self.loader.generator.set_state(self.generator.get_state())
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        self.iterated = True
        for batch, worker_id, state in self.loader:
            if worker_id is not None:
                self.worker_states[worker_id] = state
            yield batch
