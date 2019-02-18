# res2s2am

## Overview

Res2s2am implements a software pipeline for identifying functional non-coding SNP (Single Nucleotide Polymorphism). The ```Res``` stands for ResNet-based, the ```2s``` for involving the bidirectional feature of SNP function in the model, the ```2a``` for biallelic feature and ```m``` for combining the metadata into the model. The model taks encoded DNA sequences and preselected SNP annotation features as input and outputs the corresponding score of a certain SNP.

## Systems Requirements

Res2s2am currently only has a command-line based usage and aims for online GUI for future development. To run the scripts for data extraction, preprocessing and machine learning, certain system package dependencies are required as following:

absl-py==0.1.12
astor==0.6.2
beautifulsoup4==4.6.0
biopython==1.70
bleach==1.5.0
certifi==2018.1.18
chardet==3.0.4
cycler==0.10.0
decorator==4.2.1
gast==0.2.0
gevent==1.3.4
greenlet==0.4.13
grequests==0.3.0
grpcio==1.10.0
html5lib==0.9999999
httplib2==0.11.3
idna==2.6
ipython==6.2.1
ipython-genutils==0.2.0
jedi==0.11.1
joblib==0.12.0
Keras==2.1.5
kiwisolver==1.0.1
lxml==4.2.3
Markdown==2.6.11
matplotlib==2.2.2
numpy==1.14.2
pandas==0.23.3
parso==0.1.1
pexpect==4.4.0
pickleshare==0.7.4
Pillow==5.0.0
pkg-resources==0.0.0
prompt-toolkit==1.0.15
protobuf==3.5.2.post1
ptyprocess==0.5.2
Pygments==2.2.0
PyMySQL==0.8.0
pyparsing==2.2.0
pysam==0.14.1
python-dateutil==2.7.1
pytz==2018.3
PyVCF==0.6.8
PyYAML==3.12
requests==2.19.1
scikit-learn==0.19.1
scipy==1.0.1
seaborn==0.9.0
selenium==2.48.0
simplegeneric==0.8.1
six==1.11.0
SQLAlchemy==1.2.5
tensorboard==1.6.0
tensorflow==1.6.0
termcolor==1.1.0
torch==0.4.0
torchvision==0.2.0
tqdm==4.23.4
traitlets==4.3.2
urllib3==1.22
wcwidth==0.1.7
Werkzeug==0.14.1
wget==3.2

## Usage

