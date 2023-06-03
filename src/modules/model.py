"""Collection of custom neural networks."""
import torch
import torch.nn as nn

from src.config import Config
from src.modules.module import MixerBlock
from src.modules.module import ConvMixerBlock
from src.modules.module import ConvBlock
from src.modules.module import PositionEmbedding
from src.modules.module import TokenEmbedding
from src.modules.module import Classifier
from src.modules.module import ConvClassifier
from src.modules.module import TransformerBlock
from src.modules.module import PositionEmbedding
from src.modules.module import TokenEmbedding
from src.utils.tools import init_weights


import random
class MLPMixer(nn.Module):
    """Character-level isotropic MLP-Mixer."""

    def __init__(self, config: Config, num_iter: int = None):
        """Initializes MLPMixer."""
        super().__init__()
        self.num_iter = num_iter
        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionEmbedding(config)

        num_blocks = config.model.num_blocks
        mixer_blocks = [MixerBlock(config) for _ in range(num_blocks)]
        self.mixer_blocks = nn.Sequential(*mixer_blocks)
        self.classifier = Classifier(config)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        num_iter = self.num_iter or random.randint(1, 8)
        for _ in range(num_iter):
            x = self.mixer_blocks(x)
        x = self.classifier(x)
        return x


class MLPMixer_(nn.Module):
    """Character-level isotropic MLP-Mixer."""

    def __init__(self, config: Config):
        """Initializes MLPMixer."""
        super().__init__()
        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionEmbedding(config)

        num_blocks = config.model.num_blocks
        mixer_blocks = [MixerBlock(config) for _ in range(num_blocks)]
        self.mixer_blocks = nn.Sequential(*mixer_blocks)
        self.classifier = Classifier(config)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        x = self.mixer_blocks(x)
        x = self.classifier(x)
        return x


class ConvMixer(nn.Module):
    """Character-level isotropic Conv-Mixer."""

    def __init__(self, config: Config):
        """Initializes ConvMixer."""
        super().__init__()
        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionEmbedding(config)

        num_blocks = config.model.num_blocks
        mixer_blocks = [ConvMixerBlock(config) for _ in range(num_blocks)]
        self.mixer_blocks = nn.Sequential(*mixer_blocks)
        self.classifier = ConvClassifier(config)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        x = self.mixer_blocks(x)
        x = self.classifier(x)
        return x


class ConvModel(nn.Module):
    """Character-level isotropic convolutional neural network."""

    def __init__(self, config: Config):
        """Initializes convolutional neural network ."""
        super().__init__()
        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionEmbedding(config)

        num_blocks = config.model.num_blocks
        conv_blocks = [ConvBlock(config) for _ in range(num_blocks)]
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.classifier = ConvClassifier(config)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        x = self.conv_blocks(x)
        x = self.classifier(x)
        return x


class Transformer(nn.Module):
    """Isotropic multi-head self-attention transformer neural network."""

    def __init__(self, config: Config):
        """Initializes transformer module."""
        super().__init__()
        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionEmbedding(config)

        num_blocks = config.model.num_blocks
        blocks = [TransformerBlock(config) for _ in range(num_blocks)]
        self.transformer_blocks = nn.Sequential(*blocks)
        self.classifier = Classifier(config)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        x = self.transformer_blocks(x)
        x = self.classifier(x)
        return x