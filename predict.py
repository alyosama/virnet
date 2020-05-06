import time
import argparse
import pandas as pd

from NNClassifier import NeuralClassifier
from constants import c
import utils
import os
parser = argparse.ArgumentParser(description='VirNet a deep neural network model for virus identification')
parser.add_argument('--input_dim', dest='input_dim', type=int, default=500, help='input dim (default: 500)')
parser.add_argument('--input', dest='input_path', help='input file')
parser.add_argument('--output', dest='output_path', default='output.csv', help='output file csv')
parser.add_argument('--model_dir', dest='model_dir',default='data/saved_model', help='the path of the model')
args = parser.parse_args()

       
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
    df['result']=utils.predict_classes(predictions)
    df.to_csv(output_path)

def main():
    print('Starting VirNet')
    
    # Create Model
    model = NeuralClassifier(input_dim=args.input_dim, ngrams=c.MODEL.ngrams, model_dir = args.model_dir)

    ## Load Data
    df_data=utils.load_data(args.input_path)
    x_data = model.tokenize(df_data['SEQ'].values,ngrams=c.MODEL.ngrams)

    model_path = os.path.join(args.model_dir,'model_{}.h5'.format(args.input_dim))
    model.load_model(model_path)

    predictions=run_pred(model,x_data)
    save_pred(df_data,predictions,args.output_path)

if __name__ == "__main__":
    main()

