import json
import os
from typing import Union


class ModelConfig:

  model_type: str = ''

  def __init__(self, **kwargs):
    pass

  @classmethod
  def from_pretrained(cls, json_file: Union[str, os.PathLike]) -> 'ModelConfig':
    with open(json_file, 'r', encoding='utf-8') as reader:
      text = reader.read()
    config_dict = json.loads(text)

    return cls(**config_dict)
