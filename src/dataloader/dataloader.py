import numpy
import os
import pathlib
import random
import re

import torch
import torchtext
from torch.utils.data import DataLoader

from src.config.config import Config
from src.dataset import ArithmeticDataset
from src.dataset import AlgebraicDataset
from src.dataset import BooleanDataset
from src.dataset import CharDataset


def seed_worker(worker_id):
    """Seed dataloader workers."""
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(config: Config) -> DataLoader:
    """Creates dataloader for specified dataset."""

    dataset = config.dataset.dataset
    num_terms = config.dataset.num_terms
    num_workers = config.dataloader.num_workers
    batch_size = config.trainer.batch_size

    input_sequence_length = config.model.input_sequence_length
    output_sequence_length = config.model.output_sequence_length

    if dataset == "arithmetic":
        train_dataset = ArithmeticDataset(num_terms=num_terms)
    elif dataset == "algebraic":
        train_dataset = AlgebraicDataset(num_terms=num_terms)
    elif dataset == "boolean":
        train_dataset = BooleanDataset(num_terms=num_terms)
    elif dataset == "tinystories":
        data = load_tinystories()
        train_dataset = CharDataset(
            data=data,
            input_length=input_sequence_length,
            output_length=output_sequence_length,
        )
    else:
        raise NotImplementedError(f"Dataloader for {dataset} not implemented.")

    # Both models.
    if input_sequence_length == -1:
        config.model.input_sequence_length = train_dataset.max_input_length
    else:
        config.model.input_sequence_length = input_sequence_length 

    if output_sequence_length == -1:
        config.model.output_sequence_length = train_dataset.max_output_length
    else:
        config.model.output_sequence_length = output_sequence_length

    if dataset == "tinystories":
        config.data.num_tokens = train_dataset.num_tokens
        config.data.num_classes = train_dataset.num_tokens
    else:
        config.data.num_input_tokens = train_dataset.num_input_tokens
        config.data.num_output_tokens = train_dataset.num_output_tokens

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

    return train_loader


def load_tinystories() -> str:
    """Downloads and cleans TinyStories validation dataset (~19MB file) from 
    Huggingface.

    Function replaces '<|endoftext|>' token with single '<' character to 
    indicate end of story.

    Dataset can be found here: 
        https://huggingface.co/datasets/roneneldan/TinyStories

    Returns:
        Single string holding TinyStories validation dataset.
    """
    dataset_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
    file_name = dataset_url.split("/")[-1]

    # Create folder for data.
    dataset_dir = "data/tinystories/"
    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)

    # Download data if not already done.
    torchtext.utils.download_from_url(url=dataset_url, root=dataset_dir)

    cwd = os.getcwd()
    file_path = cwd + "/" + dataset_dir + file_name

    with open(file_path, mode="r") as file:
        data = file.read()

    # Replace end of story token with single character.
    data = re.sub("\<\|endoftext\|\>", "<", data)

    return data