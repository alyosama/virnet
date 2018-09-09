import os
import re
import random
import datetime
import time
import pandas as pd
import numpy as np
import argparse
import keras
from keras.models import Sequential,load_model
from Bio import SeqIO

MIN_DIM = 500

parser = argparse.ArgumentParser(description='VirNet a deep neural network model for virus identification')
parser.add_argument('--model_path', dest='model_path', default='data/model.h5', help='the path of the model')
parser.add_argument('--type', dest='input_type', default='fasta', help='choose the input type fasta, fastq or CSV')
parser.add_argument('--input', dest='input_path', help='input file')
parser.add_argument('--output', dest='output_path', default='output.csv', help='output file csv')
args = parser.parse_args()


def load_csv(file_path):
    df=pd.read_csv(file_path)
    return df

def load_fasta(file_path):
    data_list=[]
    for record in SeqIO.parse(file_path, "fasta"):
        data_list.append([record.id,str(record.seq)])
    print('Loaded {0} fragments'.format(len(data_list)))

    df=pd.DataFrame(data_list,columns=['ID','SEQ'])
    return df

def load_fastq(file_path):
    data_list=[]
    for record in SeqIO.parse(file_path, "fastq"):
        data_list.append([record.id,str(record.seq)])
    print('Loaded {0} fragments'.format(len(data_list)))

    df=pd.DataFrame(data_list,columns=['ID','SEQ'])
    return df

def load_data(input_path,input_type):
    print('Loading Data {0}'.format(input_path))
    if input_type == 'fasta':
        return load_fasta(input_path)
    elif input_type == 'fastq':
        return load_fastq(input_path)
    elif input_type == 'csv':
        return load_csv(input_path)
    else:
        print('Not supported input type {0}'.format(input_type))

def process_data(input_data):
    input_dim=len(input_data['SEQ'][0])
    print('Processing Fragments with {0}bp'.format(input_dim))
    if(input_dim < MIN_DIM):
        input_dim=MIN_DIM
    dna_dict={'A':1,'C':2,'G':3,'T':4,'N':5,' ':0}
    def decode(seq):
        new_seq=np.zeros(input_dim)
        seq=re.sub(r'[^ATGCN]','N',seq.upper())
        for i in range(len(seq[:input_dim])):
            new_seq[i]=dna_dict[seq[i]]
        return new_seq.astype(np.int)
    
    input_data['SEQ']=input_data['SEQ'].apply(decode)
    X=input_data['SEQ'].values.tolist()
    X=np.array(X).reshape(len(X),input_dim,1)

    return X
    
def predict_classes(proba):
    thresh=0.5
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > thresh).astype('int32')

def run_pred(model,input_data): 
    print('Start Predictions')
    start=time.time()
    X=process_data(input_data)
    y_prop=model.predict(X,batch_size=1024)
    end=time.time()
    print('Predicting time\t{0:.2f} seconds\n'.format(end-start))
    return y_prop

def save_pred(input_data,predictions,output_path):
    print('Saving Predictions to {0}'.format(output_path))
    df=pd.DataFrame(input_data['ID'],columns=['ID'])
    df['PROP']=predictions
    df['RESULT']=predict_classes(predictions)
    df.to_csv(output_path)

def main():
    print('Running VirNet')
    model = load_model(args.model_path)
    input_data=load_data(args.input_path,args.input_type)
    predictions=run_pred(model,input_data)
    save_pred(input_data,predictions,args.output_path)

if __name__ == "__main__":
    main()

