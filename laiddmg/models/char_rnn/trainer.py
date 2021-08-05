from argparse import ArgumentParser


from ... import logging


logger = logging.get_logger(__name__)


def train_parser(parser: ArgumentParser) -> ArgumentParser:
  if parser is None:
    parser = ArgumentParser()

  char_rnn_parser = parser.add_parser('char_rnn')

  char_rnn_parser.add_argument('--num_train_epochs',
                               default=100,
                               type=int,
                               help='number of epochs for training')
  char_rnn_parser.add_argument('--train_batch_size',
                               default=64,
                               type=int,
                               help='batch size per device for training')

  logger.info('char_rnn_parser')

  return char_rnn_parser
