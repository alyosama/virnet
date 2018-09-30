## TO CALL 
# 
# 
# 
# python predict.py --input_dim=3000 --input=../../data/3-fragments/fna/viral_test.fna_3000.fna --output=../../benchmark/vir_results/viral_test.fna_3000.fna --model_path=../../work_dir/models/saved_model/model_3000.h5

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

parser = argparse.ArgumentParser(description='VirNet a deep neural network model for virus identification')
parser.add_argument('--input_dim', dest='input_dim', type=int, default=500, help='input dim (default: 500)')
parser.add_argument('--cell_type', dest='model_name', default='lstm', help='model type which is lstm,gru,rnn (default: lstm)')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=256, help='Batch size (default: 256)')
parser.add_argument('--n_layers', dest='n_layers', type=int, default=2, help='number of layers(default: 2)')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='learning rate(default: 0.001)')
parser.add_argument('--epoch', dest='ep', type=int, default=30, help='number of epochs(default: 30)')
parser.add_argument('--patience',dest='pt',type=int, default=5, help='number of declining epochs before choosing the best epoch for saving')
parser.add_argument('--embed_size',dest='embs',type=int, default=128,help='Size of Embedding layer of input tokens (128)')
parser.add_argument('--ngrams', dest='ngrams', type=int, default=5, help='number of substring used in each sequence (3) ')
parser.add_argument('--work_dir', dest='work_dir', default='../../work_dir', help='Training Work dir')
parser.add_argument('--input', dest='input_path', help='input file')
parser.add_argument('--output', dest='output_path', default='output.csv', help='output file csv')
parser.add_argument('--model_path', dest='model_path', help='the path of the model')


######### PARAMS #############
args = parser.parse_args()
model_name=args.model_name
input_dim=args.input_dim
output_dim=1

######## FILE PATHS ##########
experiment_name='{0}_I{1}_L{2}'.format(model_name,input_dim,args.n_layers)

def load_data(data_path):
    def clean_seq(seq):
        return re.sub(r'[^ATGCN]','N',seq.upper())

    def load_csv_fragments(data_path):
        #df=pd.read_csv(data_path)
        data_list=[]
        for record in SeqIO.parse(data_path, "fasta"):
            data_list.append([record.id,str(record.seq)])
        print('Loaded {0} fragments from {1}'.format(len(data_list),data_path))

        df=pd.DataFrame(data_list,columns=['ID','SEQ'])

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
    df['PROP']=predictions
    df['RESULT']=predict_classes(predictions)
    df.to_csv(output_path)

def main():
    print('Starting Experiment {0}'.format(experiment_name))

    # Load Data
    df_data=load_data(args.input_path)

    # Create Model
    print('Loading Model')
    model = NeuralClassifier(exp_name=experiment_name, type=args.model_name, nepochs=args.ep, patience=args.pt, l_rate=args.lr,\
    batch_size=args.batch_size, embed_size=args.embs,\
    nlayers=args.n_layers, maxlen = int(math.ceil(args.input_dim * 1.0 / args.ngrams)))

    model.load_model(args.model_path)

    # Prepare data
    x_data = model.tokenize_predict(df_data['SEQ'].values,ngrams=args.ngrams)

    predictions=run_pred(model,x_data)
    save_pred(x_data,predictions,args.output_path)


if __name__ == "__main__":
    main()

