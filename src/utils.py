import re
import pandas as pd 
from Bio import SeqIO

def clean_seq(seq):
    return re.sub(r'[^ATGCN]','N',seq.upper())


def load_csv(file_path):
    df=pd.read_csv(file_path)
    return df

def load_fasta(file_path):
    data_list=[]
    for record in SeqIO.parse(file_path, "fasta"):
        data_list.append([record.id,record.description,str(record.seq)])
    print('Loaded {0} fragments'.format(len(data_list)))

    df=pd.DataFrame(data_list,columns=['ID','DESC','SEQ'])
    df['SEQ']=df['SEQ'].apply(clean_seq)
    return df

def load_fastq(file_path):
    data_list=[]
    for record in SeqIO.parse(file_path, "fastq"):
        data_list.append([record.id,record.description,str(record.seq)])
    print('Loaded {0} fragments'.format(len(data_list)))

    df=pd.DataFrame(data_list,columns=['ID','DESC','SEQ'])
    df['SEQ']=df['SEQ'].apply(clean_seq)
    return df

def load_data(input_path):
    print('Loading Data {0}'.format(input_path))
    input_type=input_path.split('.')[-1]
    if input_type in ['fasta','fna','fa']:
        return load_fasta(input_path)
    elif input_type == 'fastq':
        return load_fastq(input_path)
    elif input_type == 'csv':
        return load_csv(input_path)
    else:
        print('Not supported input type {0}'.format(input_type))
 
 
def predict_classes(proba,thresh=0.5):
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > thresh).astype('int32')