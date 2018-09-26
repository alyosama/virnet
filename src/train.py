import os
import re
import gzip
import random
import datetime
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from contextlib import redirect_stdout
from Bio import SeqIO


from sklearn.metrics import classification_report,roc_auc_score,accuracy_score,roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from NNClassifier import NeuralClassifier

parser = argparse.ArgumentParser(description='VirNet a deep neural network model for virus identification')
parser.add_argument('--mode', dest='mode',type=bool, default=False, help='if you want train mode (0) or eval mode (1) (default: 0)')
parser.add_argument('--input_dim', dest='input_dim', type=int, default=500, help='input dim (default: 500)')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=512, help='Batch size (default: 512)')
parser.add_argument('--cell_type', dest='model_name', default='lstm', help='model type which is lstm,gru,rnn (default: lstm)')
parser.add_argument('--n_layers', dest='n_layers', type=int, default=2, help='number of layers(default: 2)')
parser.add_argument('--n_neurons', dest='nn', type=int, default=128, help='number of neurons(default: 128)')
parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='learning rate(default: 0.01)')
parser.add_argument('--epoch', dest='ep', type=int, default=30, help='number of epochs(default: 30)')
parser.add_argument('--data', dest='data', default='../../data/3-fragments/fna', help='train mode (mode =0) Training and Testing data dir, eval mode (mode =1) path of test file')
parser.add_argument('--balance_data', dest='balance_data', type=bool, default=False, help='Balance data for two classes using undersampler')
parser.add_argument('--sample', dest='sample', type=bool, default=False, help='sample data (500 points) to test script')
parser.add_argument('--work_dir', dest='work_dir', default='../../work_dir', help='Training Work dir')
parser.add_argument('--model_path', dest='model_path', default='model.h5', help='in case you are in in eval model ')

######### PARAMS #############
args = parser.parse_args()
genomes=['non_viral','viral']
model_name=args.model_name
input_dim=args.input_dim
output_dim=1
data_dir=args.data


######## FILE PATHS ##########
experiment_name='{0}_I{1}_L{2}_N{3}_ep{4}_lr{5}'.format(model_name,input_dim,args.n_layers,args.nn,args.ep,args.lr)
data_file='{0}_{1}.fna_{2}.fna'
model_dir=os.path.join(args.work_dir,'models')
experiment_dir=os.path.join(args.work_dir,'experiments')
if(args.mode):
    model_path=args.model_path
else:
    model_path=os.path.join(model_dir,'model_'+experiment_name+'{epoch:02d}-{val_acc:.2f}.h5')
experiment_curve_file_path=os.path.join(experiment_dir,'{0}_roc_curve.png'.format(experiment_name))
experiment_logs_file_path=os.path.join(experiment_dir,'{0}_logs.txt'.format(experiment_name))
experiment_traincurve_file_path=os.path.join(experiment_dir,'{0}_train_curve.png'.format(experiment_name))
experiment_logits_file_path=os.path.join(experiment_dir,'{0}_logits.h5'.format(experiment_name))


############ HELPER FUNCTIONS ############
logs=[]
def create_dirs():
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

def load_data():
    def load_fasta(file_path):
        data_list=[]
        for record in SeqIO.parse(file_path, "fasta"):
            data_list.append([record.id,str(record.seq)])
        print('Loaded {0} fragments from {1}'.format(len(data_list),file_path))

        df=pd.DataFrame(data_list,columns=['ID','SEQ'])
        return df
    
    def clean_seq(seq):
        return re.sub(r'[^ATGCN]','N',seq.upper())

    def load_csv_fragments(genome,ty,input_dim):
        data_path=os.path.join(data_dir,data_file.format(genome,ty,input_dim))
        #df=pd.read_csv(data_path)
        df=load_fasta(data_path)
        df['SEQ']=df['SEQ'].apply(clean_seq)
        if genome == 'viral':
            df['LABEL']=1
        else:
            df['LABEL']=0
        return df
        
    print('Loading training and testing data')
    train_list=[]
    test_list=[]
    for genome in genomes:
        train_list.append(load_csv_fragments(genome,'train',input_dim))
        test_list.append(load_csv_fragments(genome,'test',input_dim))

    df_train=pd.concat(train_list)
    df_test=pd.concat(test_list)
    ## SHUFFLE TRAINING DATA
    df_train=df_train.sample(frac=1).reset_index(drop=True)
    
    print('Training len {0}'.format(len(df_train)))
    print('Testing len {0}'.format(len(df_test)))
    
    ### JUST FOR TESTING PURPOSE
    if(args.sample):
        n_sample=500
        print('Sample first {0} of data'.format(n_sample))
        df_train=df_train.sample(n_sample)
        df_test=df_test.sample(n_sample)
    return df_train,df_test


def under_sample_data(X_train,y_train):
    print('UnderSample Data')
    rus = RandomUnderSampler(random_state=42)
    rus.fit(X_train, y_train)
    X_train, y_train = rus.sample(X_train, y_train)
    print('New Size {0}'.format(len(X_train)))
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

def predict_classes(proba):
    thresh=0.5
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > thresh).astype('int32')

def evaluate_model(model,X_test,y_test):
    print('Evaluate model ... ')
    start=time.time()
    target_names = ['Not Virus', 'Virus']
    y_prop=model.predict(X_test)
    end=time.time()
    y_pred=predict_classes(y_prop)
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
    df_train,df_test=load_data()
    model = NeuralClassifier()
    #model=create_model(model_name,input_dim,output_dim,args.nn,args.n_layers)
    X_train,X_test = model.tokenize_set(df_train['SEQ'].values,df_test['SEQ'].values)
    y_train=df_train['LABEL'].values
    y_test=df_test['LABEL'].values

    if(args.balance_data):
        X_train,y_train=under_sample_data(X_Train,y_train)
    history = model.fit(X_train,y_train)
    plot_train(history)
    evaluate_model(model,X_test,y_test)

if __name__ == "__main__":
    main()

