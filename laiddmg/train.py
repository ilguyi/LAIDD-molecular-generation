#!/usr/bin/env/python

import os
import sys

from datetime import datetime

from laiddmg import (
  get_train_args,
  set_output_dir,
  CharRNNConfig,
  VAEConfig,
  Tokenizer,
  CharRNNModel,
  VAEModel,
  measure_duration_time,
)

from . import logging


def main():

  start_time = datetime.now()
  # get training args
  args = get_train_args()
  model_type = sys.argv[1]
  args = set_output_dir(model_type, args)
  logger = logging.get_logger(__name__, os.path.join(args.output_dir, 'out.log'))

  logger.info(args)
  logger.info(f'model type: {model_type}')
  logger.info(f'use device: {args.device}')

  assert model_type in ['char_rnn', 'vae']
  if model_type == 'char_rnn':
    config = CharRNNConfig()
    tokenizer = Tokenizer()
    model = CharRNNModel(config)
  else:
    config = VAEConfig()
    tokenizer = Tokenizer()
    model = VAEModel(config)
  print(config)
  print(tokenizer('c1cccc1c'))
  print(model)

  print(model.device)
  print(model.dtype)
  print(model.num_parameters())

  inputs = tokenizer(['c1cccc1c', 'c1ccc1c'])
  outputs, hiddens = model(**inputs)
  print(outputs.shape)
  print(hiddens[0].shape)
  print(hiddens[1].shape)

  end_time = datetime.now()
  measure_duration_time(end_time - start_time)


if __name__ == '__main__':
  main()
