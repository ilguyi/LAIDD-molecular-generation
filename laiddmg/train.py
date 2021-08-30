#!/usr/bin/env/python

import os
import sys

from datetime import datetime

from laiddmg import (
  get_train_args,
  set_output_dir,
  Tokenizer,
  CharRNNConfig,
  CharRNNModel,
  VAEConfig,
  VAEModel,
  get_rawdataset,
  get_dataset,
  TRAINER_MAPPING,
  measure_duration_time,
)

from . import logging_utils


def main():

  start_time = datetime.now()
  # get training args
  args = get_train_args()
  model_type = sys.argv[1]
  args = set_output_dir(model_type, args)
  logger = logging_utils.get_logger(__name__, os.path.join(args.output_dir, 'out.log'))

  logger.info(args)
  logger.info(f'model type: {model_type}')
  logger.info(f'use device: {args.device}')

  assert model_type in ['char_rnn', 'vae']
  tokenizer = Tokenizer()
  if model_type == 'char_rnn':
    config = CharRNNConfig()
    model = CharRNNModel(config)
  else:
    config = VAEConfig()
    model = VAEModel(config)

  print(config)
  print(tokenizer('c1ccccc1'))
  print(model)

  print(model.device)
  print(model.dtype)
  print(model.num_parameters())

  # get raw dataset (SMILES)
  train = get_rawdataset('train')

  # get PyTorch Dataset
  train_dataset = get_dataset(train, tokenizer)

  # get trainer
  trainer = TRAINER_MAPPING[model_type]
  t = trainer(model=model,
              args=args,
              train_dataset=train_dataset,
              tokenizer=tokenizer)
  t.train()

  end_time = datetime.now()
  measure_duration_time(end_time - start_time)


if __name__ == '__main__':
  main()
