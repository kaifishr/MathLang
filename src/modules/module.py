"""Common modules for neural networks."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config


class TokenEmbedding(nn.Module):
    """Token embedding module.

    Token embedding for MLP-Mixer and convolutional neural networks.

    Embeds the integer representing a character token as a vector of given 
    dimension for MLP-Mixer networks or as a square feature map for 
    convolutional networks.

    Attributes:
        embedding_dim:
    """

    def __init__(self, config: Config) -> None:
        """Initializes PositionalEmbedding."""
        super().__init__()

        num_tokens = config.data.num_tokens
        model_type = config.model.type
        embedding_dim = config.model.embedding_dim

        # Get right shape for token embedding.
        if model_type in ("mlpmixer", "transformer"):
            size = (num_tokens, embedding_dim)
        elif model_type in ("convmixer", "cnn"):
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

        model_type = config.model.type
        sequence_length = config.model.input_sequence_length
        embedding_dim = config.model.embedding_dim

        if model_type in ("mlpmixer", "transformer"):
            size = (sequence_length, embedding_dim)
        elif model_type in ("convmixer", "cnn"):
            size = (sequence_length, embedding_dim, embedding_dim)
        else:
            raise NotImplementedError(
                f"Embedding for model type '{model_type}' not implemented."
            )

        position_embedding = config.model.position_embedding

        if position_embedding.encoding == "zeros":
            embedding = torch.zeros(size=size)
        elif position_embedding.encoding == "ones":
            embedding = torch.ones(size=size)
        elif position_embedding.encoding == "random_normal":
            embedding = torch.normal(mean=0.0, std=0.02, size=size)
        else:
            raise NotImplementedError(
                f"Embedding '{position_embedding.encoding}' not implemented."
            )

        self.embedding = nn.Parameter(data=embedding, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.embedding
        return x


class MetaLinear3(torch.nn.Module):
    """Linear meta layer.

    Meta layers compute weight matrix for their linear transformation
    dynamically based on the input tensor `x`. To reduce computational costs, 
    this version of MetaLayer computes two vectors based on the input that are 
    then used to compute an outer product representing the weight matrix.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        """Initializes a meta layer instance."""
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.linear_in = torch.nn.Linear(
            in_features=in_features, 
            out_features=in_features, 
            bias=bias
        )
        
        self.linear_out = torch.nn.Linear(
            in_features=in_features, 
            out_features=out_features, 
            bias=bias
        )

        self.linear_b = torch.nn.Linear(
            in_features=in_features, 
            out_features=out_features, 
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()

        # Compute weights of weight matrix conditioned on input `x`.

        # 1) Compute weight vectors.
        w_1 = self.linear_in(x)
        w_2 = self.linear_out(x)

        # 2) Compute weight matrix as result of an outer product.
        w = torch.einsum("bsi,bsj->bsij", w_2, w_1)
        w = w.reshape(batch_size * seq_len, self.out_features, self.in_features)
        w = torch.nn.functional.layer_norm(w, normalized_shape=(self.in_features,))

        # 3) Compute bias weights.
        b = self.linear_b(x)
        b = torch.nn.functional.layer_norm(b, normalized_shape=(self.out_features,))
        b = b.reshape(batch_size * seq_len, self.out_features, 1)

        # 4) Reshape input for matrix multiplication by folding sequence 
        # dimensions into batch dimension and by adding additional dimension.
        x = x.reshape(batch_size * seq_len, embed_dim, 1)

        # 5) Transform input `x` using the dynamically computes weight matrix.
        x = torch.bmm(w, x) + b
        x = x.reshape(batch_size, seq_len, self.out_features)

        return x


class MetaLinear2(torch.nn.Module):
    """Meta linear layer class.

    This meta linear layer computes a weight matrices and biases based on the 
    input with which the linear transformation then is performed. To improve
    performance, the layer first reduces the size of the input features before 
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
                in_features=in_features, 
                out_features=hidden_features, 
                bias=bias
            ),
            torch.nn.Linear(
                in_features=hidden_features,
                out_features=in_features * out_features,
                bias=bias,
            ),
        )
        self.b_linear = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=in_features, 
                out_features=hidden_features, 
                bias=bias
            ),
            torch.nn.Linear(
                in_features=hidden_features, 
                out_features=out_features, 
                bias=bias
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
            in_features=in_features, 
            out_features=in_features * out_features, 
            bias=bias
        )
        self.b_linear = torch.nn.Linear(
            in_features=in_features, 
            out_features=out_features, 
            bias=bias
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
            # MetaLinear(in_features=dim, out_features=hidden_dim),
            # MetaLinear(in_features=hidden_dim, out_features=dim),
            nn.Linear(in_features=dim, out_features=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=dim),
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
            nn.LayerNorm([sequence_length, embedding_dim, embedding_dim]),
            nn.GELU(),
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
            nn.Conv2d(
                sequence_length, 
                sequence_length, 
                kernel_size=1
            ),
            nn.LayerNorm([sequence_length, embedding_dim, embedding_dim]),
            nn.GELU(),
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
            nn.Conv2d(
                sequence_length, 
                sequence_length, 
                kernel_size, 
                padding="same",
            ),
            nn.LayerNorm([sequence_length, embedding_dim, embedding_dim]),
            nn.GELU(),
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
            DepthwiseConvolution(config=config), 
            PointwiseConvolution(config=config)
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
                in_features=input_sequence_length, 
                out_features=output_sequence_length
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
                in_features=embedding_dim * embedding_dim, 
                out_features=num_classes
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier(x)
        return x


class Mask(nn.Module):
    """Implements a Mask module.

    Comes with different types of mask applied to the attention matrix
    of dot-products. The masks weight can be trained if required.
    """

    def __init__(self, config: Config):
        """Initializes the Mask module."""
        super().__init__()

        self.cfg_mask = config.model.mask
        self.is_activated = config.model.mask.is_activated

        if self.is_activated:
            self.max_sequence_length = config.model.input_sequence_length
            mask_type = config.model.mask.type
            size = (self.max_sequence_length, self.max_sequence_length)

            if mask_type == "trainable_additive":
                mask_params = torch.zeros(size=size)
            elif mask_type == "trainable_multiplicative":
                mask_params = torch.ones(size=size)
            else:
                raise NotImplementedError(
                    f"Mask '{mask_type}' not implemented."
                )

            self.mask = nn.Parameter(data=mask_params, requires_grad=True)

            self.mask_function = None
            self._install_mask(mask_type)

    def _install_mask(self, mask_type: str) -> None:
        # Create masks.
        if mask_type == "trainable_additive":
            self.mask_function = lambda x, seq_len: self.mask[:seq_len, :seq_len] + x
        elif mask_type == "trainable_multiplicative":
            self.mask_function = lambda x, seq_len: self.mask[:seq_len, :seq_len] * x
        else:
            raise NotImplementedError(f"Mask {mask_type} not implemented.")

    def _apply_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Applies installed mask to input tensor.

        Args:
            x: Input tensor.

        Returns: Masked tensor.
        """
        sequence_length = x.size(-1)
        # sequence_length = self.max_sequence_length # TODO: Use this and simplify.
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

        embedding_dim = config.model.embedding_dim
        self.num_heads = config.model.self_attention.num_heads

        assert embedding_dim % self.num_heads == 0

        self.head_dim = embedding_dim // self.num_heads
        self.dropout_prob = config.model.dropout_prob

        self.comp_keys = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim,
        )
        self.comp_queries = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim,
        )
        self.comp_values = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim,
        )

        self.mask = Mask(config=config)

        self.linear = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim,
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
        keys = keys.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        values = values.view(batch_size, sequence_length, self.num_heads, self.head_dim)

        # Scaled dot-product self-attention
        out = torch.einsum("bqhd,bkhd->bhqk", [queries, keys]) / embedding_dim**0.5

        out = self.mask(out)

        out = F.softmax(out, dim=-1)
        out = self.dropout(out)

        # Second part of scaled dot-product self-attention.
        out = torch.einsum("bhql,blhd->bqhd", [out, values])
        out = out.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)

        # Unify all heads in linear transformation.
        out = self.linear(out)
        out = self.dropout(out)

        return out


class TransformerBlock(nn.Module):
    """Module consisting of self-attention and full connected layers."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        embedding_dim = config.model.embedding_dim 
        expansion_factor = config.model.expansion_factor
        dropout_prob = config.model.dropout_prob

        self.attention = MultiHeadSelfAttention(config)

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, int(expansion_factor * embedding_dim)),
            nn.Dropout(dropout_prob),
            nn.GELU(),
            nn.Linear(int(expansion_factor * embedding_dim), embedding_dim),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))
        return x