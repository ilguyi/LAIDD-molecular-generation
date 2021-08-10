# flake8: noqa
# There's no way to ignore F401 imported but unused warnings in this module
# but to preserve other warnings. So, don't check this module at all.


__version__ = '0.0.1'


from collections import OrderedDict

# Parser
from .common_parser import (
  get_train_args,
  # get_generate_args,
)

# Configs
from .models.char_rnn.configuration import CharRNNConfig
from .models.vae.configuration import VAEConfig

# Tokenizer
from .tokenization_utils import Tokenizer

# Models
from .models.char_rnn.modeling import CharRNNModel
from .models.vae.modeling import VAEModel

# Trainers
from .models.char_rnn.char_rnn_trainer import CharRNNTrainer
from .models.vae.vae_trainer import VAETrainer

# Datasets
from .datasets import get_rawdataset
from .datasets import get_dataset

# Utils
from .utils import (
  set_output_dir,
  measure_duration_time,
)


TRAINER_MAPPING = OrderedDict(
  [
    ('char_rnn', CharRNNTrainer),
    ('vae', VAETrainer),
  ]
)
