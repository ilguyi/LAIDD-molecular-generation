import argparse
# from abc import abstractmethod
# from typing import Dict, Union, Any, Optional
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter

from .tokenization_utils import Tokenizer
from .utils import (
  set_seed,
  args_and_config_to_json_files,
)


from . import logging


logger = logging.get_logger(__name__)


class Trainer:

  def __init__(
    self,
    model: nn.Module = None,
    args: argparse.Namespace = None,
    train_dataset: Dataset = None,
    tokenizer: Optional[Tokenizer] = None,
    optimizer: optim.Optimizer = None,
    scheduler: optim.lr_scheduler = None,
    **kwargs,
  ):
    self.model = model
    self.args = args
    self.train_dataset = train_dataset
    self.tokenizer = tokenizer
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.global_step = 0

    set_seed(self.args.seed)
    if model is None:
      raise RuntimeError('`Trainer requires a `model` arguments.')

    logger.info('`training from scratch`')

    # tb_log_dir = os.path.join(args.output_dir, 'logs')
    # self.tb_writer = SummaryWriter(tb_log_dir)
    # logger.info(f'Created tensorboard writer in {tb_log_dir}.')

    if self.args.device == torch.device('cuda:0'):
      logger.info('Use one gpu')
    else:
      logger.info('Use only cpu')
    self.model = self.model.to(self.args.device)

    # get train_dataloader
    self.train_dataloader = self.get_train_dataloader()
    # add num_train_steps_per_epoch to args
    self.args.num_training_steps_per_epoch = len(self.train_dataloader)
    self.args.num_training_steps = len(self.train_dataloader) * args.num_train_epochs
    logger.info(f'the number of training steps per epoch: {self.args.num_training_steps_per_epoch}')
    logger.info(f'the total number of training steps per epoch: {self.args.num_training_steps}')

    args_and_config_to_json_files(self.args, self.model.config)

  def get_train_dataloader(self) -> DataLoader:
    if self.train_dataset is None:
      raise ValueError('Trainer: training requires a `train_dataset`.')

    return DataLoader(
        self.train_dataset,
        batch_size=self.args.train_batch_size,
        shuffle=True,
        # collate_fn=self._collate_fn,
        num_workers=16,
    )
