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


## Getting Started

### Prerequisites

* `python` >= 3.7
* [`pytorch`](https://pytorch.org) >= 1.7
* `numpy`, `pandas`
* `rdkit`
  * 이 패키지를 설치할때 `rdkit`은 자동으로 설치되지 않아서 따로 설치를 해야 합니다.
  * `rdkit` install manual: [https://www.rdkit.org/docs/Install.html](https://www.rdkit.org/docs/Install.html)

##### `rdkit` 설치 방법 (`conda`이용 하여 설치를 추천)
```bash
$ conda install -c conda-forge rdkit
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
$ git clone https://github.com/ilguyi/LAIDD-molecule-generation.git
$ cd molgen
$ pip install .
```


## Quickstart
