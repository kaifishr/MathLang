"""Holds methods for Tensorboard.
"""
import math

import matplotlib.pyplot as plt
import numpy
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.modules.module import TokenEmbedding
from src.modules.module import PositionEmbedding


def add_graph(
    model: nn.Module, dataloader: DataLoader, writer: SummaryWriter, config: dict
) -> None:
    """Add graph of model to Tensorboard.

    Args:
        model:
        dataloader:
        writer:
        config:
    """
    device = config.trainer.device
    x_data, _ = next(iter(dataloader))
    writer.add_graph(model=model, input_to_model=x_data.to(device))


def add_position_embedding_weights(
    writer: SummaryWriter, model: nn.Module, global_step: int
) -> None:
    """Adds visualization of position embeddings Tensorboard."""
    for name, module in model.named_modules():
        if isinstance(module, PositionEmbedding):
            embedding = module.embedding.detach().cpu()
            x_min = torch.min(embedding)
            x_max = torch.max(embedding)
            embedding = (embedding - x_min) / (x_max - x_min)

            if len(embedding.shape) == 3:
                dataformats = "NCHW"
                embedding = embedding.unsqueeze(dim=1)
            elif len(embedding.shape) == 2:
                dataformats = "HW"

            writer.add_image(name, embedding, global_step, dataformats=dataformats)


def add_token_embedding_weights(
    writer: SummaryWriter, model: nn.Module, global_step: int
) -> None:
    """Adds visualization of token embeddings to Tensorboard."""
    for name, module in model.named_modules():
        if isinstance(module, TokenEmbedding):
            embedding = module.embedding.detach().cpu()
            x_min = torch.min(embedding)
            x_max = torch.max(embedding)
            embedding = (embedding - x_min) / (x_max - x_min)

            if len(embedding.shape) == 3:
                dataformats = "NCHW"
                embedding = embedding.unsqueeze(dim=1)
            elif len(embedding.shape) == 2:
                dataformats = "HW"

            writer.add_image(name, embedding, global_step, dataformats=dataformats)


def add_linear_weights_(
    writer: SummaryWriter, model: nn.Module, global_step: int, n_samples_max: int = 128
) -> None:
    """Adds visualization of channel and token embeddings to Tensorboard."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu()

            height, width = weight.shape
            dim = int(math.sqrt(width))

            if not dim**2 == width:
                continue

            # Extract samples
            n_samples = min(height, n_samples_max)
            weight = weight[:n_samples]

            # Rescale
            x_min, _ = torch.min(weight, dim=-1, keepdim=True)
            x_max, _ = torch.max(weight, dim=-1, keepdim=True)
            weight = (weight - x_min) / (x_max - x_min + 1e-6)

            # Reshape
            weight = weight.reshape(-1, 1, dim, dim)

            writer.add_images(name, weight, global_step, dataformats="NCHW")


def add_linear_weights(
    writer: SummaryWriter,
    model: nn.Module,
    global_step: int,
    n_samples_max: int = 32,
    do_rescale: bool = False,
) -> None:
    """Adds visualization of channel and token embeddings to Tensorboard."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu().numpy()

            height, width = weight.shape
            dim = int(math.sqrt(width))

            if not dim**2 == width:
                continue

            # Extract samples
            n_samples = min(height, n_samples_max)
            weight = weight[:n_samples]

            # Rescale
            if do_rescale:
                x_min = numpy.min(weight, axis=-1, keepdims=True)
                x_max = numpy.max(weight, axis=-1, keepdims=True)
                weight = (weight - x_min) / (x_max - x_min + 1e-6)

            # Reshape
            weight = weight.reshape(-1, dim, dim)

            # Plot weights
            ncols = 8
            nrows = int(math.ceil(n_samples / ncols))
            figsize = (0.5 * ncols, 0.5 * nrows)

            figure, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=figsize,
                layout="constrained",
            )

            for ax, w in zip(axes.flatten(), weight):
                ax.imshow(w, cmap="bwr", interpolation="none")  # none, spline16, ...

            for ax in axes.flatten():
                ax.axis("off")

            writer.add_figure(name, figure, global_step)


def add_kernel_weights(
    writer: SummaryWriter,
    model: nn.Module,
    global_step: int,
    num_samples_max: int = 64,
) -> None:
    """Adds visualization of channel and token embeddings to Tensorboard."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.detach().cpu().numpy()

            num, channels, height, width = weight.shape

            if (height == 1) and (width == 1):
                continue

            if channels != 1:
                continue

            # Extract samples
            num_samples = min(num, num_samples_max)
            weight = weight[:num_samples]

            # Switch axes
            weight = weight.transpose(0, 2, 3, 1)

            # Plot weights
            ncols = 8
            nrows = int(math.ceil(num_samples / ncols))
            figsize = (0.5 * ncols, 0.5 * nrows)

            figure, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=figsize,
                layout="constrained",
            )
            for ax, w in zip(axes.flatten(), weight):
                ax.imshow(w, cmap="bwr", interpolation="none")  # none, spline16, ...

            for ax in axes.flatten():
                ax.axis("off")

            writer.add_figure(name, figure, global_step)
