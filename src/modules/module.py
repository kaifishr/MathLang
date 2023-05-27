"""Common modules for neural networks."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class TokenEmbedding(nn.Module):
    """Token embedding module.

    Embeds an integer as a vector of defined dimension.

    Attributes:
        max_sequence_length:
        embedding_dim:
    """

    def __init__(self, config: Config) -> None:
        """Initializes PositionalEmbedding."""
        super().__init__()

        num_tokens = config.data.num_tokens

        n_heads = config.transformer.self_attention.n_heads
        head_dim = config.transformer.self_attention.head_dim
        embedding_dim = n_heads * head_dim

        self.cfg_token_embedding = config.transformer.token_embedding
        size = (num_tokens, embedding_dim)

        if self.cfg_token_embedding.encoding == "random_normal":
            embedding = torch.normal(mean=0.0, std=0.01, size=size)
        elif self.cfg_token_embedding.encoding == "sinusoidal":
            embedding = self._sinusoidal_encoding(size=size)
        else:
            raise NotImplementedError(
                f"Embedding {self.cfg_token_embedding.encoding} not implemented."
            )

        requires_grad = True if self.cfg_token_embedding.is_trainable else False
        self.embedding = nn.Parameter(data=embedding, requires_grad=requires_grad)

    @staticmethod
    def _sinusoidal_encoding(size: tuple) -> torch.Tensor:
        """Sinusoidal encoding scheme.

        See also: https://arxiv.org/abs/1706.03762
        """
        num_tokens, embedding_dim = size
        position = torch.arange(num_tokens).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        encoding = torch.zeros(num_tokens, embedding_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        return encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Receives sequences of token identifiers and returns embedding.

        Args:
            x: Integer tensor holding integer token identifiers.

        Returns:
            Embedded tokens.
        """
        # x = self.embedding(x)  # TODO: use this later with nn.Embedding
        x = self.embedding[x]  # TODO: Test. Seems to work as well.
        # x = F.embedding(x, self.embedding)  # TODO: Test.
        return x


class PositionEmbedding(nn.Module):
    """Positional embedding module.

    Positional embedding with different encoding schemes.

    Attributes:
        max_sequence_length:
        embedding_dim:
    """

    def __init__(self, config: Config) -> None:
        """Initializes PositionalEmbedding."""
        super().__init__()
        max_sequence_length = config.transformer.max_sequence_length
        n_heads = config.transformer.self_attention.n_heads
        head_dim = config.transformer.self_attention.head_dim
        embedding_dim = n_heads * head_dim

        self.pos_emb = config.transformer.position_embedding
        self.is_activated = config.transformer.position_embedding.is_activated

        if self.is_activated:
            requires_grad = True if self.pos_emb.is_trainable else False
            size = (max_sequence_length, embedding_dim)

            if self.pos_emb.encoding == "zeros":
                embedding = torch.zeros(size=size)
            elif self.pos_emb.encoding == "ones":
                embedding = torch.ones(size=size)
            elif self.pos_emb.encoding == "random_normal":
                embedding = torch.normal(mean=0.0, std=0.01, size=size)
            elif self.pos_emb.encoding == "sinusoidal":
                embedding = self._sinusoidal_encoding(size=size)
            else:
                raise NotImplementedError(
                    f"Embedding {self.pos_emb.encoding} not implemented."
                )

            self.embedding = nn.Parameter(data=embedding, requires_grad=requires_grad)

    @staticmethod
    def _sinusoidal_encoding(size: tuple) -> torch.Tensor:
        """Sinusoidal encoding scheme.

        See also: https://arxiv.org/abs/1706.03762
        """
        max_sequence_length, embedding_dim = size
        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        encoding = torch.zeros(max_sequence_length, embedding_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_activated:
            # pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)#.unsqueeze(0)
            # print(f"{self.embedding[pos].shape = }")
            # print(f"{self.embedding[:x.size(1)].shape = }")
            sequence_length = x.size(1)
            x = x + self.embedding[:sequence_length]
        return x


class Mask(nn.Module):
    """Implements a Mask module.

    Comes with different types of mask applied to the attention matrix
    of dot-products. The masks weight can be trained if required.
    """

    def __init__(self, config: Config):
        """Initializes the Mask module."""
        super().__init__()

        self.cfg_mask = config.transformer.mask
        self.is_activated = config.transformer.mask.is_activated

        if self.is_activated:
            self.max_sequence_length = config.transformer.max_sequence_length
            size = (self.max_sequence_length, self.max_sequence_length)

            mask_type = config.transformer.mask.type

            # Create masks.
            if mask_type == "trainable_additive":
                self.mask = nn.Parameter(
                    data=torch.zeros(size=size), requires_grad=True
                )
            elif mask_type == "trainable_multiplicative":
                self.mask = nn.Parameter(data=torch.ones(size=size), requires_grad=True)
            elif mask_type == "causal":
                self.mask = nn.Parameter(
                    data=torch.tril(input=torch.ones(size=size)),
                    requires_grad=False,
                )
            else:
                raise NotImplementedError(f"Mask {mask_type} not implemented.")
            # TODO: self.mask -> self.weight?

            self.mask_function = None
            self._install_mask(mask_type)

    def _install_mask(self, mask_type: str) -> None:
        # Create masks.
        if mask_type == "trainable_additive":
            self.mask_function = lambda x, seq_len: self.mask[:seq_len, :seq_len] + x
        elif mask_type == "trainable_multiplicative":
            self.mask_function = lambda x, seq_len: self.mask[:seq_len, :seq_len] * x
        elif mask_type == "causal":
            self.mask_function = lambda x, seq_len: x.masked_fill(
                self.mask[:seq_len, :seq_len] == 0, float("-inf")
            )
        else:
            raise NotImplementedError(f"Mask {mask_type} not implemented.")

    def _apply_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Applies installed mask to input tensor.

        Args:
            x: Input tensor.

        Returns: Masked tensor.
        """
        sequence_length = x.size(-1)
        x = self.mask_function(x, sequence_length)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_activated:
            x = self._apply_mask(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Implements multi-head self-attention for image data."""

    def __init__(self, config: Config) -> None:
        """Initializes multi-head self-attention module."""
        super().__init__()

        cfg = config.transformer.self_attention

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout_prob = cfg.dropout_prob
        self.use_bias = cfg.use_bias
        # self.use_mask = cfg.use_mask

        embedding_dim = cfg.n_heads * cfg.head_dim
        bias = True if self.use_bias else False

        self.comp_keys = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim, bias=bias
        )
        self.comp_queries = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim, bias=bias
        )
        self.comp_values = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim, bias=bias
        )

        # Trainable mask. Let the network decide how the mask should look like.
        self.mask = Mask(config=config)

        self.linear = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim, bias=bias
        )

        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        batch_size, sequence_length, embedding_dim = x.size()

        # Compute keys, queries, and values over all embedding vectors.
        keys = self.comp_keys(x)
        queries = self.comp_queries(x)
        values = self.comp_values(x)

        # Split keys, queries, and values for processing in different heads.
        keys = keys.view(batch_size, sequence_length, self.n_heads, self.head_dim)
        queries = queries.view(batch_size, sequence_length, self.n_heads, self.head_dim)
        values = values.view(batch_size, sequence_length, self.n_heads, self.head_dim)

        # Scaled dot-product self-attention
        out = torch.einsum("bqhd,bkhd->bhqk", [queries, keys]) / embedding_dim**0.5

        out = self.mask(out)

        out = F.softmax(out, dim=-1)
        out = self.dropout(out)

        # Second part of scaled dot-product self-attention.
        out = torch.einsum("bhql,blhd->bqhd", [out, values])
        out = out.reshape(batch_size, sequence_length, self.n_heads * self.head_dim)

        # Unify all heads in linear transformation.
        out = self.linear(out)
        out = self.dropout(out)

        return out


class TransformerBlock(nn.Module):
    """Module consisting of self-attention and full connected layers."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        cfg_block = config.transformer.transformer_block
        hidden_expansion = cfg_block.hidden_expansion
        dropout_prob = cfg_block.dropout_prob

        cfg_attention = config.transformer.self_attention
        embedding_dim = cfg_attention.n_heads * cfg_attention.head_dim

        self.attention = MultiHeadSelfAttention(config)

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, int(hidden_expansion * embedding_dim)),
            nn.Dropout(dropout_prob),
            nn.GELU(),
            nn.Linear(int(hidden_expansion * embedding_dim), embedding_dim),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))
        return x