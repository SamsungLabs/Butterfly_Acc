# Main part is obtained from huggingface: https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/fnet/configuration_fnet.py
# We modify the model definition to support hybrid FNet

""" FNet model configuration """

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

FNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/fnet-base": "https://huggingface.co/google/fnet-base/resolve/main/config.json",
    "google/fnet-large": "https://huggingface.co/google/fnet-large/resolve/main/config.json"
    # See all FNet models at https://huggingface.co/models?filter=fnet
}


class BflyLR_FNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FNetModel`]. It is used to
    instantiate an FNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the FNet [fnet-base](https://huggingface.co/google/fnet-base) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model
    outputs. Read the documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the FNet model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`FNetModel`] or
            [`TFFNetModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 4):
            The vocabulary size of the `token_type_ids` passed when calling [`FNetModel`] or
            [`TFFNetModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_tpu_fourier_optimizations (`bool`, *optional*, defaults to `False`):
            Determines whether to use TPU optimized FFTs. If `True`, the model will favor axis-wise FFTs
            transforms. Set to `False` for GPU/CPU hardware, in which case n-dimensional FFTs are used.
        tpu_short_seq_length (`int`, *optional*, defaults to 512):
            The sequence length that is expected by the model when using TPUs. This will be used to initialize the DFT
            matrix only when *use_tpu_fourier_optimizations* is set to `True` and the input sequence is shorter
            than or equal to 4096 tokens.
    Example:
    ```python
    >>> from transformers import FNetModel, FNetConfig
    >>> # Initializing a FNet fnet-base style configuration
    >>> configuration = FNetConfig()
    >>> # Initializing a model from the fnet-base style configuration
    >>> model = FNetModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bflylr_fnet"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=4,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_tpu_fourier_optimizations=False,
        tpu_short_seq_length=512,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        ###############################
        # Hybrid FNet config
        num_attention_layers=2,
        attention_layout="Bottom", # One of "Bottom", "Top", "Middle"
        num_attention_heads=12,
        attention_probs_dropout_prob=0.1,
        position_embedding_type="absolute", # Need to double check with flax implementation: https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.SelfAttention.html
        ###############################
        # Butterfly Low Rank config
        lr_ratio=0.1,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_tpu_fourier_optimizations = use_tpu_fourier_optimizations
        self.tpu_short_seq_length = tpu_short_seq_length
        self.num_attention_layers = num_attention_layers
        self.attention_layout = attention_layout
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob =attention_probs_dropout_prob
        self.position_embedding_type = position_embedding_type
        self.lr_ratio = lr_ratio