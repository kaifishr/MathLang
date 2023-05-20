"""Common modules for neural networks."""
import torch
import torch.nn as nn

from src.config import Config


class TokenEmbedding(nn.Module):
    """Token embedding module.

    Token embedding for MLP-Mixer and convolutional neural networks.

    Embeds the integer representing a character token as a vector of given dimension
    for MLP-Mixer networks or as a square feature map for convolutional networks.

    Attributes:
        embedding_dim:
    """

    def __init__(self, config: Config) -> None:
        """Initializes PositionalEmbedding."""
        super().__init__()

        num_tokens = config.data.num_tokens
        model_type = config.model.type
        embedding_dim = config.model.embedding_dim

        if model_type == "mlpmixer":
            size = (num_tokens, embedding_dim)
        elif model_type == "convmixer" or model_type == "cnn":
            size = (num_tokens, embedding_dim, embedding_dim)
        else:
            raise NotImplementedError(
                f"Embedding for model type {model_type} not implemented."
            )

        embedding = torch.normal(mean=0.0, std=0.02, size=size)
        self.embedding = nn.Parameter(data=embedding, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Receives sequences of token identifiers and returns embedding.

        Args:
            x: Integer tensor holding integer token identifiers.

        Returns:
            Embedded tokens.
        """
        x = self.embedding[x]
        return x


class PositionEmbedding(nn.Module):
    """Positional embedding module.

    Attributes:
        sequence_length:
        embedding_dim:
    """

    def __init__(self, config: Config) -> None:
        """Initializes PositionalEmbedding."""
        super().__init__()

        sequence_length = config.model.input_sequence_length
        model_type = config.model.type
        embedding_dim = config.model.embedding_dim

        if model_type == "mlpmixer":
            size = (sequence_length, embedding_dim)
        elif model_type == "convmixer" or model_type == "cnn":
            size = (sequence_length, embedding_dim, embedding_dim)
        else:
            raise NotImplementedError(
                f"Embedding for model type {model_type} not implemented."
            )

        embedding = torch.normal(mean=0.0, std=0.02, size=size)
        self.embedding = nn.Parameter(data=embedding, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.embedding
        return x


class MetaLinear2(torch.nn.Module):
    """Meta linear layer class.

    The meta linear layer computes a weight matrices and biases
    based on the input with which the linear transformation
    then is performed.

    This layer first reduces the size of the input features before
    computing the weight matrices.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_expansion: float = 0.125,
        bias: bool = True,
    ):
        """Initializes meta layer."""
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        hidden_features = int(hidden_expansion * in_features)
        self.w_linear = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=in_features, out_features=hidden_features, bias=bias
            ),
            torch.nn.Linear(
                in_features=hidden_features,
                out_features=in_features * out_features,
                bias=bias,
            ),
        )
        self.b_linear = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=in_features, out_features=hidden_features, bias=bias
            ),
            torch.nn.Linear(
                in_features=hidden_features, out_features=out_features, bias=bias
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.size()

        # Compute weight matrix weights.
        w = self.w_linear(x)
        w = w.reshape(batch_size * sequence_length, self.out_features, self.in_features)
        w = torch.nn.functional.layer_norm(w, normalized_shape=(self.in_features,))

        # Compute bias weights.
        b = self.b_linear(x)
        b = torch.nn.functional.layer_norm(b, normalized_shape=(self.out_features,))
        b = b.reshape(batch_size * sequence_length, self.out_features, 1)

        # Reshape input for matrix multiplication.
        x = x.reshape(batch_size * sequence_length, embedding_dim, 1)

        # Compute vector-matrix multiplication with predicted parameters.
        x = torch.bmm(w, x) + b
        x = x.reshape(batch_size, sequence_length, self.out_features)

        return x


class MetaLinear(torch.nn.Module):
    """Meta linear layer class.

    The meta linear layer computes a weight matrices and biases
    based on the input with which the linear transformation
    then is performed.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """Initializes meta layer."""
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.w_linear = torch.nn.Linear(
            in_features=in_features, out_features=in_features * out_features, bias=bias
        )
        self.b_linear = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.size()

        # Compute weight matrix weights.
        w = self.w_linear(x)
        w = w.reshape(batch_size * sequence_length, self.out_features, self.in_features)
        w = torch.nn.functional.layer_norm(w, normalized_shape=(self.in_features,))

        # Compute bias weights.
        b = self.b_linear(x)
        b = torch.nn.functional.layer_norm(b, normalized_shape=(self.out_features,))
        b = b.reshape(batch_size * sequence_length, self.out_features, 1)

        # Reshape input for matrix multiplication.
        x = x.reshape(batch_size * sequence_length, embedding_dim, 1)

        # Compute vector-matrix multiplication with predicted parameters.
        x = torch.bmm(w, x) + b
        x = x.reshape(batch_size, sequence_length, self.out_features)

        return x


class MlpBlock(nn.Module):
    def __init__(self, dim: int, config: Config) -> None:
        super().__init__()

        expansion_factor = config.model.expansion_factor

        hidden_dim = expansion_factor * dim

        self.mlp_block = nn.Sequential(
            MetaLinear(in_features=dim, out_features=hidden_dim),
            MetaLinear(in_features=hidden_dim, out_features=dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_block(x)


class SwapAxes(nn.Module):
    def __init__(self, axis0: int, axis1):
        super().__init__()
        self.axis0 = axis0
        self.axis1 = axis1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.swapaxes(x, axis0=self.axis0, axis1=self.axis1)


class MixerBlock(nn.Module):
    """MLP Mixer block

    Mixes channel and token dimension one after the other.
    """

    def __init__(self, config: Config):
        super().__init__()

        sequence_length = config.model.input_sequence_length
        embedding_dim = config.model.embedding_dim

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            SwapAxes(axis0=-2, axis1=-1),
            MlpBlock(dim=sequence_length, config=config),
            SwapAxes(axis0=-2, axis1=-1),
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            MlpBlock(dim=embedding_dim, config=config),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class DepthwiseConvolution(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, config: Config):
        super().__init__()

        sequence_length = config.model.input_sequence_length
        embedding_dim = config.model.embedding_dim
        kernel_size = config.model.kernel_size

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                sequence_length,
                sequence_length,
                kernel_size,
                groups=sequence_length,
                padding="same",
            ),
            nn.GELU(),
            nn.LayerNorm([sequence_length, embedding_dim, embedding_dim]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.depthwise_conv(x)
        return x


class PointwiseConvolution(nn.Module):
    """Pointwise convolution."""

    def __init__(self, config: Config):
        super().__init__()

        sequence_length = config.model.input_sequence_length
        embedding_dim = config.model.embedding_dim

        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(sequence_length, sequence_length, kernel_size=1),
            nn.GELU(),
            nn.LayerNorm([sequence_length, embedding_dim, embedding_dim]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pointwise_conv(x)
        return x


class Convolution(nn.Module):
    """Standard convolution."""

    def __init__(self, config: Config):
        super().__init__()

        sequence_length = config.model.input_sequence_length
        embedding_dim = config.model.embedding_dim
        kernel_size = config.model.kernel_size

        self.conv_block = nn.Sequential(
            nn.Conv2d(sequence_length, sequence_length, kernel_size, padding="same"),
            nn.GELU(),
            nn.LayerNorm([sequence_length, embedding_dim, embedding_dim]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv_block(x)
        return x


class ConvBlock(nn.Module):
    """Convolutional block."""

    def __init__(self, config: Config):
        super().__init__()

        self.conv_block = nn.Sequential(
            Convolution(config=config), Convolution(config=config)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        return x


class ConvMixerBlock(nn.Module):
    """ConvMixer block."""

    def __init__(self, config: Config):
        super().__init__()

        self.conv_mixer_block = nn.Sequential(
            DepthwiseConvolution(config=config), PointwiseConvolution(config=config)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_mixer_block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, config: Config) -> None:
        """Initializes Classifier class."""
        super().__init__()

        input_sequence_length = config.model.input_sequence_length
        output_sequence_length = config.model.output_sequence_length
        embedding_dim = config.model.embedding_dim
        num_classes = config.data.num_tokens

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            SwapAxes(axis0=-2, axis1=-1),
            nn.Linear(
                in_features=input_sequence_length, out_features=output_sequence_length
            ),
            SwapAxes(axis0=-2, axis1=-1),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier(x)
        return x


class ConvClassifier(nn.Module):
    def __init__(self, config: Config) -> None:
        """Initializes Classifier class."""
        super().__init__()

        input_sequence_length = config.model.input_sequence_length
        output_sequence_length = config.model.output_sequence_length
        embedding_dim = config.model.embedding_dim
        kernel_size = config.model.kernel_size
        num_classes = config.data.num_tokens

        self.classifier = nn.Sequential(
            nn.Conv2d(
                input_sequence_length,
                output_sequence_length,
                kernel_size,
                padding="same",
            ),
            nn.Flatten(start_dim=2, end_dim=-1),
            nn.Linear(
                in_features=embedding_dim * embedding_dim, out_features=num_classes
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier(x)
        return x
