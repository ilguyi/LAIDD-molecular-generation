from ...configuration_utils import ModelConfig
from ... import logging_utils


logger = logging_utils.get_logger(__name__)


class VAEConfig(ModelConfig):

  model_type: str = 'vae'

  def __init__(
    self,
    tokenizer: str = 'moses',
    vocab_size: int = 30,
    embedding_dim: int = 30,
    encoder_hidden_dim: int = 256,
    encoder_num_layers: int = 1,
    encoder_dropout: float = 0.5,
    latent_dim: int = 128,
    decoder_hidden_dim: int = 512,
    decoder_num_layers: int = 3,
    decoder_dropout: float = 0.0,
    padding_value: int = 0,
    **kwargs,
  ):

    self.tokenizer = tokenizer
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.encoder_hidden_dim = encoder_hidden_dim
    self.encoder_num_layers = encoder_num_layers
    self.encoder_dropout = encoder_dropout
    self.latent_dim = latent_dim
    self.decoder_hidden_dim = decoder_hidden_dim
    self.decoder_num_layers = decoder_num_layers
    self.decoder_dropout = decoder_dropout
    self.padding_value = padding_value
