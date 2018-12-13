import os
import re
import random
import datetime
import math
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from Bio import SeqIO

from sklearn.metrics import classification_report,roc_auc_score,accuracy_score,roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from NNClassifier import NeuralClassifier
from constants import c

parser = argparse.ArgumentParser(description='VirNet a deep neural network model for virus identification')
parser.add_argument('--input_dim', dest='input_dim', type=int, default=500, help='input dim (default: 500)')
parser.add_argument('--input', dest='input_path', help='input file')
parser.add_argument('--output', dest='output_path', default='output.csv', help='output file csv')
parser.add_argument('--model_path', dest='model_path',default='data/saved_model/model_{}.h5', help='the path of the model')
args = parser.parse_args()


def load_data(data_path):
    def clean_seq(seq):
        return re.sub(r'[^ATGCN]','N',seq.upper())

    def load_csv_fragments(data_path):
        #df=pd.read_csv(data_path)
        data_list=[]
        for record in SeqIO.parse(data_path, "fasta"):
            data_list.append([record.id,record.description,str(record.seq)])
        print('Loaded {0} fragments from {1}'.format(len(data_list),data_path))

        df=pd.DataFrame(data_list,columns=['ID','DESC','SEQ'])

        df['SEQ']=df['SEQ'].apply(clean_seq)
        return df
        
    print('Loading Data')
    df_test = load_csv_fragments(data_path)
    
    return df_test

def predict_classes(proba):
    thresh=0.5
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > thresh).astype('int32')
   
def run_pred(model,input_data): 
    print('Start Predictions')
    start=time.time()
    y_prop=model.predict(input_data)
    end=time.time()
    print('Predicting time\t{0:.2f} seconds\n'.format(end-start))
    return y_prop

def save_pred(input_data,predictions,output_path):
    print('Saving Predictions to {0}'.format(output_path))
    df=pd.DataFrame(input_data['ID'],columns=['ID'])
    df['DESC']=input_data['DESC']
    df['score']=predictions
    df['result']=predict_classes(predictions)
    df.to_csv(output_path)

def main():
    print('Starting VirNet')

    # Create Model
    print('Loading Model')
    model = NeuralClassifier(input_dim=args.input_dim,ngrams=c.MODEL.ngrams)

    ## Load Testing Data
    df_data=load_data(args.input_path)
    x_data = model.tokenize(df_data['SEQ'].values,ngrams=c.MODEL.ngrams)


    model.load_model(args.model_path.format(args.input_dim))

    predictions=run_pred(model,x_data)
    save_pred(df_data,predictions,args.output_path)


if __name__ == "__main__":
    main()

