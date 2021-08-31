# Molecular generation for LAIDD Lecture

* **이 저장소는 한국제약바이오협회 인공지능신약개발지원센터에서 제공하는
LAIDD 강의의 일환으로 제작되었습니다.**


신약개발에서 원하는 특성을 가진 새로운 화합물을 찾아내는 것은
중요하고도 어려운 작업 입니다.
합성 가능한 분자의 공간은
<img src="https://render.githubusercontent.com/render/math?math=10^60">
으로 엄청나게 방대하여
이렇게 큰 공간에서 원하는 화합물을 찾아내는 것은 매우 어렵습니다.
그렇기 때문에 마치 넓은 바다의 모래사장에서 바늘 찾기라는 비유를 하기도 합니다.

최근 생성모델(Deep generative models)의 급속한 발전으로
이미지 데이터, 텍스트 데이터, 그래프 데이터 등 다양한 데이터들을
실제와 비슷하게 만들어 내는데 성공하고 있습니다.
ML(Machine Leanring) field에서 만들어진 생성모델들을 Chemisty field에
적용하여 de novo generation 분야에서 좋은 결과를 만들어 냈습니다.

이 강의는 de novo molecular generation 모델의 기본 모델이라 할 수 있는
RNN(Recurrent Neural Networks) 모델과 VAE(Variational AutoEncoders) 모델을
알아보고 직접 구현해보는 것을 목표로 합니다.
두 모델 모두 SMILES 데이터를 기반으로 만들어졌습니다.
SMILES(Simplified molecular-input line-entry system)는 화합물을 특정 문법에 의거하여
text(ASCII code) 기반 seqeunce 데이터로 표현하는 방법입니다.
RNN모델은 이러한 seqeunce 데이터를 잘 다룰 수 있어서 SMILES기반 생성모델
또는 QSAR모델등에 사용될 수 있습니다.

* Final update: 2021. 08. 31.
* All right reserved @ 이일구 (Il Gu Yi) 2021


## Getting Started

### Prerequisites

* `python` >= 3.7
* [`pytorch`](https://pytorch.org) >= 1.7
* `numpy`, `pandas`, `matplotlib`
* `jupyter`, `easydict`
* `rdkit`
  * 이 패키지를 설치할때 `rdkit`은 자동으로 설치되지 않아서 따로 설치를 해야 합니다.
  * `rdkit` install manual: [https://www.rdkit.org/docs/Install.html](https://www.rdkit.org/docs/Install.html)

##### `rdkit` 설치 방법 (`conda`이용 하여 설치를 추천)
```bash
$ conda install -c conda-forge rdkit=2021.03.1
```


## Installation

### 가상환경 만들기

이 패키지는 [`anaconda`](https://anaconda.org/) 환경에서 실행하는 것을 추천합니다.
먼저 `conda`를 이용하여 가상환경을 만듭니다.
```bash
$ conda create --name laiddmg python=3.7
$ conda activate laiddmg
```

`git clone`을 통해 이 패키지를 다운 받습니다.
그 후 `pip install .`으로 패키지 설치를 합니다.
```bash
$ git clone https://github.com/ilguyi/LAIDD-molecular-generation.git
$ cd LAIDD-molecular-generation
$ pip install .
$ conda install -c conda-forge rdkit=2021.03.1  # 패키지 설치시 rdkit은 설치되지 않아 따로 설치해야 합니다.
```


## Quickstart

### Jupyter notebook

간단하게 모든 과정을 하나씩 실행해 볼 수 있게
jupyter notebook 형태의 파일을 준비했습니다.
[`jupyter_char_rnn.ipynb`](https://github.com/ilguyi/LAIDD-moleculra-generation/blob/main/laiddmg/jupyter_char_rnn.ipynb),
[`jupyter_vae.ipynb`](https://github.com/ilguyi/LAIDD-moleculra-generation/blob/main/laiddmg/jupyter_vae.ipynb)
파일은 각각 CharRNN모델, ChemicalVAE모델을 실행할 수 있습니다.
Jupyter 파일 역시 이 저장소를 설치해야 이용할 수 있습니다.


### Command execution

이 github 저장소를 clone 받고 패키지를 설치하면 두가지 command가 생성됩니다.
* `laiddmg-train` command: 학습 데이터를 받아 각 모델(`CharRNN`, `ChemicalVAE`)들을 학습시키는 명령어입니다.
* `laiddmg-generate` command: 최종 학습된 모델을 불러와 새로운 분자를 생성하는 명령어입니다.

#### Training

* 스크립트 파일을 직접 참고하시면 됩니다.
* [`train.char_rnn.sh`](https://github.com/ilguyi/LAIDD-moleculra-generation/blob/main/laiddmg/train.char_rnn.sh)
* [`train.vae.sh`](https://github.com/ilguyi/LAIDD-moleculra-generation/blob/main/laiddmg/train.vae.sh)

```bash
#!/bin/bash

laiddmg-train char_rnn --seed 219 \
                       --output_dir exp1 \
                       --dataset_path ../datasets \
                       --log_steps 10 \
                       --num_train_epochs 10 \
                       --train_batch_size 128 \
                       --[model_depend_arguments] \
                       ...
```

* `seed`: 재현성을 위한 random seed number
* `output_dir`: output directory 경로
* `dataset_path`: dataset 경로
* `log_steps`: logging 하는 주기 (step 단위)
* `num_train_epochs`: 최대 학습 epoch
* `train_batch_size`: 학습 배치 사이즈
* `model_depend_arguments`: 모델에 따라 다른 training arguments

#### Generating

* generate 스크립트를 직접 참고 하시면 됩니다.
* [`generate.sh`](https://github.com/ilguyi/LAIDD-moleculra-generation/blob/main/laiddmg/generate.sh)

```bash
#!/bin/bash

laiddmg-generate char_rnn --seed 219 \
                          --checkpoint_dir outputs/char_rnn/exp1 \
                          --weights_name ckpt_100.pt \
                          --num_generation 10000 \
                          --batch_size_for_generation 256
```

* `seed`: 재현성을 위한 random seed number
* `checkpoint_dir`: load할 weights 파일이 있는 ckeckpoint directory 경로
* `weights_name`: load할 weights 파일 이름
* `num_generation`: 생성할 SMILES 갯수
* `batch_size_for_generation`: 생성할 때 배치 사이즈


### Simple python code

#### Training

```python
>>> from laiddmg import Tokenizer, CharRNNConfig, CharRNNModel, TRAINER_MAPPING

>>> model_type = 'char_rnn'
>>> tokenizer = Tokenizer()
>>> config = CharRNNConfig()
>>> model = CharRNNModel(config)

>>> inputs = tokenizer('c1ccccc1')
>>> outputs = model(**inputs)

>>> train = get_rawdataset('train')
>>> train_dataset = get_dataset(train, tokenizer)

>>> trainer = TRAINER_MAPPING[model_type]
>>> t = trainer(model=model,
...             args=args,
...             train_dataset=train_dataset,
...             tokenizer=tokenizer)
>>> t.train()
```

#### Generating

```python
>>> model.eval()
>>> outputs = model.generate(tokenizer=tokenizer,
...                          max_length=128,
...                          num_return_sequence=256,
...                          skip_special_tokens=True)
>>> print(outputs)
```

## Dataset

이 저장소에서는
molecular generation 분야에서 대표적인 벤치마크 셋인
[Molecular Sets (MOSES)](https://github.com/molecularsets/moses)
데이터 셋을 이용합니다.
이 데이터 셋은 [ZINC](https://zinc.docking.org/) 데이터 셋을 기본으로하여
[몇가지 규칙](https://github.com/molecularsets/moses#dataset)에 의거하여 필터링한 데이터 셋입니다.
원래는 총 190만개의 SMILES데이터로 구성되어 있지만
이 저장소에서는 MOSES 데이터 셋에서 random 샘플링을 하여 `train set`:`test set`의 갯수를
각각 25만, 3만으로 만들었습니다.
나중에 MOSES dataset을 이용하여 트레이닝 해보시는 것을 추천 드립니다.


## Model architectures

1. **[CharRNN]**: Charter-level recurrent neural networks / [Generating Focused Molecule Libraries for Drug Discovery with Recurrent Neural Networks](https://pubs.acs.org/doi/10.1021/acscentsci.7b00512), by Marwin H. S. Segler, Thierry Kogej, Christian Tyrchan, and Mark P. Waller.
1. **[ChemicalVAE]**: Variational autoencoders / [Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules](https://pubs.acs.org/doi/10.1021/acscentsci.7b00572), by Rafael Gómez-Bombarelli, Jennifer N. Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D. Hirzel, Ryan P. Adams, and Alán Aspuru-Guzik.


## Lecture notes

* [Lecture 1](https://www.dropbox.com/s/um8oukzoqlioff6/molecule%20generation%201.pdf?dl=0)
* [Lecture 2](https://www.dropbox.com/s/okpnpjx2wmzioyo/molecule%20generation%202.pdf?dl=0)


## Author

이일구 (Il Gu Yi)
