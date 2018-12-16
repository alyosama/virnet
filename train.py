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
import utils
from constants import c

parser = argparse.ArgumentParser(description='VirNet a deep neural network model for virus identification')
parser.add_argument('--input_dim', dest='input_dim', type=int, default=500, help='input dim (default: 500)')
parser.add_argument('--cell_type', dest='model_name', default='lstm', help='model type which is lstm,gru,rnn (default: lstm)')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=256, help='Batch size (default: 256)')
parser.add_argument('--n_layers', dest='n_layers', type=int, default=2, help='number of layers(default: 2)')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='learning rate(default: 0.001)')
parser.add_argument('--epoch', dest='ep', type=int, default=30, help='number of epochs(default: 30)')
parser.add_argument('--patience',dest='pt',type=int, default=5, help='number of declining epochs before choosing the best epoch for saving')
parser.add_argument('--embed_size',dest='embed_size',type=int, default=128,help='Size of Embedding layer of input tokens (128)')
parser.add_argument('--ngrams', dest='ngrams', type=int, default=5, help='number of substring used in each sequence (3) or number of operations used for bpe ')
parser.add_argument('--balance_data', dest='balance_data', type=bool, default=True, help='Balance data for two classes using undersampler (True) ')
parser.add_argument('--sample', dest='sample', type=int, default=-1, help='sample data (n=500 points) to test script (-1) ')
parser.add_argument('--data', dest='data', default='../../data/3-fragments/fna', help='train mode  Training and Testing data dir')
parser.add_argument('--work_dir', dest='work_dir', default='../../work_dir', help='Training Work dir')

######### PARAMS #############
args = parser.parse_args()
genomes=['non_viral','viral']
model_name=args.model_name
data_dir=args.data

c.MODEL.seq_type=args.model_name
c.MODEL.n_layers=args.n_layers
c.MODEL.nclasses=1
c.MODEL.ngrams=args.ngrams
c.MODEL.embed_size=args.embed_size
c.TRAINING.patience=args.pt
c.TRAINING.batch_size=args.batch_size
c.TRAINING.l_rate=args.lr
c.TRAINING.nepochs=args.ep


######## FILE PATHS ##########
experiment_name='{0}_I{1}_L{2}'.format(model_name,args.input_dim,args.n_layers)
data_file='{0}_{1}.fna_{2}.fna'
experiment_dir=os.path.join(args.work_dir,'experiments')
experiment_curve_file_path=os.path.join(experiment_dir,'{0}_roc_curve.png'.format(experiment_name))
experiment_logs_file_path=os.path.join(experiment_dir,'{0}_logs.txt'.format(experiment_name))
experiment_traincurve_file_path=os.path.join(experiment_dir,'{0}_train_curve.png'.format(experiment_name))
experiment_logits_file_path=os.path.join(experiment_dir,'{0}_logits.h5'.format(experiment_name))


############ HELPER FUNCTIONS ############
def create_dirs():
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

def load_data():
    def load_csv_fragments(genome,ty,input_dim):
        data_path=os.path.join(data_dir,data_file.format(genome,ty,input_dim))
        #df=pd.read_csv(data_path)
        df=utils.load_fasta(data_path)
        df['SEQ']=df['SEQ'].apply(utils.clean_seq)
        if genome == 'viral':
            df['LABEL']=1
        else:
            df['LABEL']=0
        return df
        
    print('Loading training and testing data')
    train_list=[]
    test_list=[]
    for genome in genomes:
        train_list.append(load_csv_fragments(genome,'train',args.input_dim))
        test_list.append(load_csv_fragments(genome,'test',args.input_dim))

    df_train=pd.concat(train_list)
    df_test=pd.concat(test_list)

    ## SHUFFLE TRAINING DATA
    df_train=df_train.sample(frac=1).reset_index(drop=True)  
    print('Training len {0}'.format(len(df_train)))
    print('Testing len {0}'.format(len(df_test)))
    
    return df_train,df_test

### JUST FOR TESTING or HYPERPARAMS OPTIMIZATION
def sample_data(df,n_sample):
    print('Sampling {0} of data'.format(n_sample))
    return df.sample(n_sample,random_state=42)

def balance_classes(X_train,y_train):
    print('UnderSample Data - Balance Classes')
    rus = RandomUnderSampler(random_state=42)
    rus.fit(X_train, y_train)
    X_train, y_train = rus.sample(X_train, y_train)

    print('After Balancing the new size is {0}'.format(len(X_train)))
    return X_train,y_train

def plot_train(history):
    plt.subplot(2, 1, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(experiment_traincurve_file_path)
    

def plot_roc_curve(y_test,y_prop):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_test, y_prop)
    roc_auc[0] = auc(fpr[0], tpr[0])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_prop.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
            lw=lw, label='ROC-AUC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.legend(loc="lower right")
    plt.savefig(experiment_curve_file_path)


def evaluate_model(model,X_test,y_test):
    print('Evaluate model ... ')
    logs=[]
    start=time.time()
    target_names = ['Not Virus', 'Virus']
    y_prop=model.predict(X_test)
    end=time.time()
    y_pred=utils.predict_classes(y_prop)
    logs.append(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))+'\n')
    logs.append('ROC-AUC:\t{0:.2f}\n'.format(roc_auc_score(y_test, y_prop)))
    logs.append('Accuracy:\t{0:.2f}%\n'.format(accuracy_score(y_test, y_pred)*100))
    logs.append('Classification Report:\n{0}\n'.format(classification_report(y_test, y_pred, target_names=target_names)))
    logs.append('Predicting time\t{0:.2f} sec\n'.format(end-start))
    plot_roc_curve(y_test,y_pred)
    print(''.join(logs))
    with open(experiment_logs_file_path,'w') as f:
        f.write(''.join(logs))
    #np.save(experiment_logits_file_path, y_prop)

def main():
    print('Starting Experiment {0}'.format(experiment_name))
    create_dirs()

    # Load Data
    df_train,df_test=load_data()
    if(args.sample>0):
        TEST_RATIO=0.2
        df_train=sample_data(df_train,args.sample)
        df_test=sample_data(df_test,int(args.sample*TEST_RATIO))


    # Create Model
    print('Loading Model')
    model = NeuralClassifier(input_dim=args.input_dim,ngrams=c.MODEL.ngrams)

    # Prepare data
    X_train,X_test = model.tokenize_set(df_train['SEQ'].values,df_test['SEQ'].values,ngrams=c.MODEL.ngrams)
    y_train=df_train['LABEL'].values
    y_test=df_test['LABEL'].values

    if(args.balance_data):
        X_train,y_train=balance_classes(X_train,y_train)
    
    n_viruses=len(y_train[y_train==1])
    n_pro=len(y_train[y_train==0])
    print('Viruses {0}\t Non Viruses {1}'.format(n_viruses,n_pro))


    # Train
    history = model.fit(X_train,y_train)

    # Plot History
    plot_train(history)

    #Evaluate
    evaluate_model(model,X_test,y_test)

if __name__ == "__main__":
    main()

