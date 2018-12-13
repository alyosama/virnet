import os
import random
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.metrics import classification_report,roc_auc_score,accuracy_score,roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from NNClassifier import NeuralClassifier
from constants import c
import utils

parser = argparse.ArgumentParser(description='VirNet a deep neural network model for virus identification')
parser.add_argument('--input_dim', dest='input_dim', type=int, default=500, help='input dim (default: 500)')
parser.add_argument('--input', dest='input_path', help='input file')
parser.add_argument('--output', dest='output_path', default='output.csv', help='output file csv')
parser.add_argument('--model_path', dest='model_path',default='data/saved_model/model_{}.h5', help='the path of the model')
args = parser.parse_args()
       
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
    if 'DESC' in input_data.columns:
        df['DESC']=input_data['DESC']
    df['score']=predictions
    df['result']=predict_classes(predictions)
    df.to_csv(output_path)

def main():
    print('Starting VirNet')

    # Create Model
    model = NeuralClassifier(input_dim=args.input_dim,ngrams=c.MODEL.ngrams)

    ## Load Testing Data
    df_data=utils.load_data(args.input_path)
    x_data = model.tokenize(df_data['SEQ'].values,ngrams=c.MODEL.ngrams)


    model.load_model(args.model_path.format(args.input_dim))

    predictions=run_pred(model,x_data)
    save_pred(df_data,predictions,args.output_path)


if __name__ == "__main__":
    main()

