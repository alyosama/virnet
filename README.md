# VirNet
VirNet: A deep attention model for viral reads identification

This tool is able to identifiy viral sequences from a mixture of viral and bacterial sequences. Also, it can purify viral metagenomic data from bacterial contamination

## Dependencies
Python 3.6, Tensorflow, Keras, Pandas and BioPython

## Installation

To download and install the package 

```
git clone https://github.com/alyosama/virnet
cd virnet
pip3 install requiremnts.txt

```

## Usage

The input of VirNet is the fasta file, and the output is a .csv file containing scores and prediction for each read.
you can have to specify the input dimention you want to work with flag --input_dim {100,500,1000 or 3000}. 

```
python predict.py --input_dim=500 --input=data/test/data.fna --output=output.csv

```

## For Re-Training

```
python train.py --input_dim=<n> --data=<data_folder> --work_dir=<work_dir>
```

## Conference 
http://www.icces.org.eg