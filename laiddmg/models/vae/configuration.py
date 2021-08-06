from ... import logging


logger = logging.get_logger(__name__)


class VAEConfig():

  model_type: str = 'vae'

  def __init__(
    self,
    tokenizer: str = 'moses',
    vocab_size: int = 30,
    embedding_dim: int = 30,
    hidden_dim: int = 768,
    num_layers: int = 3,
    dropout: float = 0.2,
    padding_value: int = 0,
    **kwargs,
  ):

    self.tokenizer = tokenizer
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.dropout = dropout
    self.padding_value = padding_value
