#!/usr/bin/env python

import os
import sys
from datetime import datetime
import pandas as pd

from laiddmg import (
  get_generate_args,
  set_output_dir_for_generation,
  get_batch_size_list_for_generate,
  measure_duration_time,
  CharRNNConfig,
  VAEConfig,
  Tokenizer,
  CharRNNModel,
  VAEModel,
)

from . import logging


def main():

  start_time = datetime.now()
  # get training args
  args = get_generate_args()
  model_type = sys.argv[1]
  args = set_output_dir_for_generation(model_type, args)
  logger = logging.get_logger(__name__, os.path.join(args.output_dir, 'out.log'))

  logger.info(f'args: {args}')
  logger.info(f'model type: {model_type}')
  logger.info(f'use device: {args.device}')

  # get tokenizer, config, and model
  assert model_type in ['char_rnn', 'vae']
  if model_type == 'char_rnn':
    config = CharRNNConfig.from_pretrained(os.path.join(f'{args.checkpoint_dir}', 'config.json'))
    tokenizer = Tokenizer()
    model = CharRNNModel.from_pretrained(config, os.path.join(f'{args.checkpoint_dir}', f'{args.weights_name}'))
  else:
    config = VAEConfig.from_pretrained(os.path.join(f'{args.checkpoint_dir}', 'config.json'))
    tokenizer = Tokenizer()
    model = VAEModel.from_pretrained(config, os.path.join(f'{args.checkpoint_dir}', f'{args.weights_name}'))

  logger.info(f'model type: {config.model_type}')
  logger.info(f'model config: {config}')
  logger.info(f'tokenizer vocab: {tokenizer.vocab}')
  logger.info(f'model: {model}')
  logger.info(f'model device: {model.device}')
  logger.info(f'model dtype: {model.dtype}')

  model.to(args.device)
  logger.info(f'generate on device: {model.device}')
  model.eval()

  batch_size_list = get_batch_size_list_for_generate(args)
  generated_smiles = []
  for bs in batch_size_list:
    outputs = model.generate(tokenizer=tokenizer,
                             max_length=args.max_length,
                             num_return_sequences=bs,
                             skip_special_tokens=True)
    generated_smiles += outputs

  savefile_path = os.path.join(args.output_dir, 'generated_smiles.csv')
  logger.info(f'file path to save: {savefile_path}')
  pd.DataFrame({'smiles': generated_smiles}).to_csv(
    savefile_path, header=True, index=False
  )

  end_time = datetime.now()
  measure_duration_time(end_time - start_time)


if __name__ == '__main__':
  main()
