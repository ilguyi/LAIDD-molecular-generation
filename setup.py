import re
from setuptools import find_packages
from setuptools import setup


def find_version(path):
  with open(path, 'r', encoding='utf-8') as stream:
    return re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        stream.read(),
        re.M,
    ).group(1)


def read_long_description(path):
  with open(path, 'r', encoding='utf-8') as stream:
    return stream.read()


# @see https://python-packaging.readthedocs.io/en/latest/index.html  # noqa
# @see https://setuptools.readthedocs.io/en/latest/setuptools.html#new-and-changed-setup-keywords  # noqa
# @see https://packaging.python.org/guides/packaging-namespace-packages/#native-namespace-packages  # noqa
setup(
    name='LAIDD-mol-gen',
    version=find_version('laiddmg/__init__.py'),
    author='Il Gu Yi',
    author_email='ilgu.yi.219@gmail.com',
    url='https://github.com/ilguyi/LAIDD-molecule-generation',
    license='MIT',
    description='Molecular generation modules for LAIDD Lecture',
    long_description=read_long_description('README.md'),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
      'Environment :: GPU :: NVIDIA CUDA :: 11.0',
      'License :: OSI Approved :: MIT License',
      'Operating System :: OS Independent',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.7',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=[''],
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
      # LIST THE DEPENDENCIES OF YOUR PACKAGE
      # FOR INSTANCE,
      # 'numpy>=1.19.2,
      'torch>=1.0',
      'numpy',
      'pandas',
      # 'scikit-learn',
      # 'scipy',
      'flake8',
      'pre-commit',
      'tensorboard',
      'tqdm',
    ],
    entry_points={
      'console_scripts': [
        'laiddmg-train = laiddmg.train:main',
        'laiddmg-generate = laiddmg.generate:main',
        'laiddmg-eval= laiddmg.eval:main',
      ]
    },
)
