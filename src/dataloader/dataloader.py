import numpy
import random
import torch
from torch.utils.data import DataLoader

from src.config.config import Config
from src.dataset import ArithmeticDataset


def seed_worker(worker_id):
    """Seed dataloader workers."""
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(config: Config) -> tuple[DataLoader, DataLoader]:
    """Creates dataloader for specified dataset."""

    dataset = config.dataloader.dataset
    num_workers = config.dataloader.num_workers
    batch_size = config.trainer.batch_size

    if dataset == "arithmetic":

        train_dataset = ArithmeticDataset()
        valid_dataset = train_dataset
        config.model.input_sequence_length = train_dataset.max_input_length
        config.model.output_sequence_length = train_dataset.max_output_length
        config.data.num_classes = train_dataset.num_tokens  # TODO: Check for redundancy
        config.data.num_tokens = train_dataset.num_tokens

    elif dataset == "algebraic":
        raise NotImplementedError(f"Dataloader for {dataset} not implemented.")
    elif dataset == "boolean":
        raise NotImplementedError(f"Dataloader for {dataset} not implemented.")
    else:
        raise NotImplementedError(f"Dataloader for {dataset} not implemented.")

    generator = torch.Generator()
    generator.manual_seed(config.random_seed)

    if "cuda" in str(config.trainer.device):
        pin_memory = True
    else:
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=2 * batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader