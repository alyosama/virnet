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

## Accepted Conference 
http://www.icces.org.eg



## Reference and Citation
please cite this paper, if you use our method:
[Abdelkareem, A., Khalil, M. I., Elaraby. M., Abbas, H. M., & Elbehery, A. H. (2018). VirNet: a deep attention model for viral reads identification, 13th IEEE International Conference on Computer Engineering and Systems (ICCES 2018) ](https://github.com/alyosama/virnet/blob/master/documents/224_CR.pdf##)


## Copyright
Copyright (C) 2018 Faculty of Engineering, Ain Shams University
Author: Aly O. Abdelkareem


## License
This program is freely available and Commercial users should contact Eng. Aly at aly.osama@eng.asu.edu.eg
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
