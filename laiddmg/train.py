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
  else:
    config = VAEConfig()
    tokenizer = Tokenizer()
  print(config)
  print(tokenizer('c1cccc1c'))

  end_time = datetime.now()
  measure_duration_time(end_time - start_time)


if __name__ == '__main__':
  main()
