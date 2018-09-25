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
from keras.layers.recurrent import LSTM
#from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint,EarlyStopping
import tensorflow as tf
from keras import callbacks
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential 
from keras import backend as K 
from keras.layers.core import *
import numpy as np
from AttentionLayer import AttentionWeightedAverage

class NeuralClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,attention=True,nclasses=1,nepochs = 10,type='lstm',patience = 2,batch_size = 256,embeddings = None,embed_size = 30,vocab_size = None,maxlen = 167,nlayers = 1,ngpus = 1,val_set = 0.05):
        """
        Called when initializing the classifier
        """
        self.ngpus = ngpus
        self.maxlen = maxlen
        self.nclasses = nclasses
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.file_path="../../work_dir/models/weights_base.best.hdf5"
        self.val_size = val_set
        self.model = None
        self.embeddings = embeddings 
        self.embed_size = embed_size
        if self.embeddings is not None:
            self.embed_size = self.embeddings.embedding_size
        self.vocab_size = vocab_size
        self.attention = attention
        self.patience = patience
        self.type=  type
        self.name = type
        self.nlayers = nlayers
        self.callbacks_list = []
        self.tokenizer = None

    def lstm_model(self):

        inp = Input(shape=(self.maxlen,) )
        if self.embeddings is not None: 
            self.vocab_size = self.embeddings.embedding_matrix.shape[0]
            x = Embedding(self.vocab_size, self.embed_size, weights=[self.embeddings.embedding_matrix],trainable=False)(inp)
        else:
            x = Embedding(self.vocab_size, self.embed_size)(inp)
        for i in range(self.nlayers):
            x = LSTM(self.embed_size, return_sequences=True, dropout=0.1, recurrent_dropout=0.1 )(x)
        if self.attention:
            x = AttentionWeightedAverage()(x)
        else:
            x = GlobalMaxPool1D()(x)
        x = Dense(self.embed_size, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(self.nclasses, activation="sigmoid")(x)
        if self.ngpus>1:
            with tf.device("/cpu:0"):
                model = Model(inputs=inp, outputs=x)
            #model = multi_gpu_model(model, gpus=self.ngpus)
        else:
            model = Model(inputs=inp, outputs=x)
            
        adam=optimizers.Adam()
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        return model

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        checkpoint = ModelCheckpoint(self.file_path, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only = True, mode='min')
        early = EarlyStopping(monitor="val_loss", mode="min", patience=self.patience)
        train = X
        train_label = y
        self.callbacks_list = [early,checkpoint] #early
        self.model = self.lstm_model()
        if self.val_size==0:
            self.callbacks_list = []
        history = self.model.fit(train, train_label, batch_size=self.batch_size, epochs=self.nepochs, validation_split=self.val_size,callbacks =self.callbacks_list ,shuffle=True)
        return history


    def predict(self, X):
        #self.model.load_weights(self.file_path)
        return self.model.predict([X],batch_size=1024, verbose=1)

    def predict_proba(self, X):
        return self.predict(X)

    def score(self, X=None, y=None):
        # counts number of values bigger than mean
        if X is None and y is None:
            return roc_auc_score(self.val_label,self.model.predict(self.val))
        else:
            return roc_auc_score(y,self.model.predict(X))
    
    def word_break(self,sentences,ngrams=3):
        if ngrams==0:
            return sentences
        result = [ re.sub(r'(.{'+str(ngrams)+'})',r'\1 ',sent).strip() for sent in sentences]
        return result
      
    def tokenize_train(self,train_sentences):
        if self.vocab_size is not None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size,char_level=False)
        else:
            self.tokenizer = Tokenizer(char_level=False)
        self.tokenizer.fit_on_texts(self.word_break(train_sentences))
        self.vocab_size = len(self.tokenizer.word_index)
        if self.embeddings is not None:
            self.embeddings.set_embeddings_matrix(self.tokenizer.word_index,self.vocab_size)

    def tokenize(self,sentences):
        list_tokenized = self.tokenizer.texts_to_sequences(self.word_break(sentences))
        return pad_sequences(list_tokenized , maxlen=self.maxlen)

    def tokenize_set(self,train_sentences,test_sentences): 
        self.tokenize_train(list(train_sentences)+list(test_sentences))
        X_t =self.tokenize(list(train_sentences))
        X_te = self.tokenize(list(test_sentences))
        return X_t,X_te