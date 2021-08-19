# Copyright 2020 Qptuna, Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# modified by Il Gu Yi, 2021
# original source code is at
# `https://github.com/huggingface/transformers/tree/master/src/transformers/utils/logging.py`
""" Logging utilities. """


import logging
import os
import sys
import threading

from typing import Optional


_lock = threading.Lock()
_stream_handler: str = None
_file_handler: str = None
library_root_logger: str = None


log_levels = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}

_default_log_level = log_levels['info']


def _get_library_name() -> str:

  return __name__.split('.')[0]


def _get_library_root_logger() -> logging.Logger:

  return logging.getLogger(_get_library_name())


def _configure_library_root_logger(log_path: str = None) -> None:

  global _stream_handler, _file_handler, library_root_logger

  formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s %(asctime)s >> %(message)s')

  with _lock:

    if _stream_handler is not None:
      pass
    else:
      _stream_handler = logging.StreamHandler()
      _stream_handler.flush = sys.stderr.flush

      library_root_logger = _get_library_root_logger()
      library_root_logger.setLevel(_default_log_level)

      library_root_logger.addHandler(_stream_handler)
      library_root_logger.propagate = False

      _stream_handler.setFormatter(formatter)

    if log_path is not None:
      if _file_handler is not None:
        raise ValueError(f'{log_path} must be one file.')
      else:
        log_dir = os.path.dirname(log_path)
        if log_dir != '':
          os.makedirs(log_dir, exist_ok=True)
          _file_handler = logging.FileHandler(log_path)
          library_root_logger.addHandler(_file_handler)

        _file_handler.setFormatter(formatter)


def get_logger(name: Optional[str] = None,
               log_path: str = None) -> logging.Logger:

  if name is None:
    name = _get_library_name()

  _configure_library_root_logger(log_path)

  return logging.getLogger(name)
