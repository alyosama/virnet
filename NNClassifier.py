# -*- coding: utf8 -*-
from sklearn.base import BaseEstimator, ClassifierMixin
from keras import optimizers
import re
from sklearn.metrics import roc_auc_score
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,SpatialDropout1D, Activation,GRU,TimeDistributed, MaxPooling1D, Convolution1D,Conv1D
from keras.layers.merge import Concatenate,Multiply
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate,BatchNormalization,Lambda
from keras.models import Model
from keras.layers import merge
from keras.layers.recurrent import LSTM,GRU,RNN
from keras.callbacks import ModelCheckpoint,EarlyStopping
import tensorflow as tf
import pickle
from keras import callbacks
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential 
from keras import backend as K 
from keras.layers.core import *
import numpy as np
import math
import os
from AttentionLayer import AttentionWeightedAverage
from constants import c

class NeuralClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,embeddings = None,vocab_size = None,input_dim=500,ngrams=5,  model_dir = 'data/saved_model'):
        """
        Called when initializing the classifier
        """
        self.input_dim=input_dim
        self.maxlen =  int(math.ceil(input_dim * 1.0 / ngrams))
        self.model = None
        self.embeddings = embeddings 
        self.vocab_size = vocab_size
        self.callbacks_list = []
        self.tokenizer = None
    
        self.checkpoint_path="models/model-{val_loss:.2f}.h5"
        self.tokenize_path = os.path.join(model_dir,'tokenizer_{}.pkl'.format(self.input_dim))

    def lstm_model(self):

        inp = Input(shape=(self.maxlen,) )
        if self.embeddings is not None: 
            self.vocab_size = self.embeddings.embedding_matrix.shape[0]
            x = Embedding(self.vocab_size, c.MODEL.embed_size, weights=[self.embeddings.embedding_matrix],trainable=False)(inp)
        else:
            x = Embedding(self.vocab_size,c.MODEL.embed_size)(inp)
        for i in range(c.MODEL.n_layers):
            if(c.MODEL.seq_type=='lstm'):
                x = LSTM(c.MODEL.embed_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1 )(x)
            elif(c.MODEL.seq_type=='gru'):
                x = GRU(c.MODEL.embed_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1 )(x)
            else:
                x = RNN(c.MODEL.embed_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1 )(x)
        if c.MODEL.attention:
            x = AttentionWeightedAverage()(x)
        else:
            x = GlobalMaxPool1D()(x)
        x = Dense(c.MODEL.embed_size, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(c.MODEL.nclasses, activation="sigmoid")(x)
        
        #if self.ngpus>1:
        #    with tf.device("/cpu:0"):
        #        model = Model(inputs=inp, outputs=x)
        #else:
        #    model = Model(inputs=inp, outputs=x)

        model=Model(inputs=inp, outputs=x)   
        adam=optimizers.Adam(lr=c.TRAINING.l_rate)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        return model

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only = True, mode='min')
        early = EarlyStopping(monitor="val_loss", mode="min", patience=c.TRAINING.patience)
        self.callbacks_list = [early,checkpoint] #early
        self.model = self.lstm_model()
        if c.TRAINING.val_size==0:
            self.callbacks_list = []
        history = self.model.fit(X, y, batch_size=c.TRAINING.batch_size, epochs=c.TRAINING.nepochs, \
        validation_split=c.TRAINING.val_size, callbacks = self.callbacks_list ,shuffle=True, verbose=2)
        return history


    def predict(self, X):
        return self.model.predict([X],batch_size=1024, verbose=1)

    def load_model(self,model_path):
        self.model = self.lstm_model()
        self.model.load_weights(model_path)

    def predict_proba(self, X):
        return self.predict(X)

    def score(self, X, y):
        # counts number of values bigger than mean
        return roc_auc_score(y,self.model.predict(X))
    
    def word_break(self,sentences,ngrams):
        if ngrams==0:
            return sentences

        result = [ re.sub(r'(.{'+str(ngrams)+'})',r'\1 ',sent).strip() for sent in sentences]
        return result
      
    def tokenize_train(self,train_sentences,ngrams):
        if self.vocab_size is not None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size,char_level=False)
        else:
            self.tokenizer = Tokenizer(char_level=False)
        self.tokenizer.fit_on_texts(self.word_break(train_sentences,ngrams))
        self.vocab_size = len(self.tokenizer.word_index)   
        self.tokenize_save()

        if self.embeddings is not None:
            self.embeddings.set_embeddings_matrix(self.tokenizer.word_index,self.vocab_size)

    def tokenize_save(self):
        
        with open(self.tokenize_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

    def tokenize_load(self):
        print('Loading Tokenizer')
        with open(self.tokenize_path, 'rb') as f:
            self.tokenizer=pickle.load(f)
            self.vocab_size = len(self.tokenizer.word_index)

    def tokenize(self,sentences,ngrams=5):
        if self.tokenizer is None:
            self.tokenize_load()
        list_tokenized = self.tokenizer.texts_to_sequences(self.word_break(sentences,ngrams))
        return pad_sequences(list_tokenized , maxlen=self.maxlen)
        
    def tokenize_set(self,train_sentences,test_sentences,ngrams=5): 	   
        self.tokenize_train(list(train_sentences)+list(test_sentences),ngrams)
        X_t =self.tokenize(list(train_sentences),ngrams)	            #train bpe model with input noperations which should be around 32k
        X_te = self.tokenize(list(test_sentences),ngrams)
        return X_t,X_te