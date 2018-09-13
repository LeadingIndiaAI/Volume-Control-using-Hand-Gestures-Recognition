# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 18:22:39 2018

@author: SG1944
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:23:30 2018

@author: SG1944
"""

import numpy as np
import os
import pandas as pd
import numpy as np
import pickle

with open(r"/home/dgxuser103/Surbhi/9/Surbhi/data/No Ges.txt","rb") as fp:
  frame1=pickle.load(fp)
with open(r"/home/dgxuser103/Surbhi/9/Surbhi/data/swip up.txt","rb") as fp:
  frame2=pickle.load(fp)
with open(r"/home/dgxuser103/Surbhi/9/Surbhi/data/swip down.txt","rb") as fp:
  frame3=pickle.load(fp)

x1=len(frame1) 
frame1=np.array(frame1)

x2=len(frame2)   
frame2=np.array(frame2)

x3=len(frame3)   
frame3=np.array(frame3)

total_length=x2+x1+x3


label=np.ones((total_length,),dtype = int)
label[0:x1-1]= 0
label[x1:x1+x2-1] = 1
label[x1+x2:]=2

X_tr_array=np.append(frame1,frame2,axis=0)
X_tr_array=np.append(X_tr_array,frame3,axis=0)


train_data = [X_tr_array,label]
(train_set, y_train) = (train_data[0],train_data[1])
print(y_train.shape)
print(train_set.shape)

from keras.utils import np_utils, generic_utils

print('X_Train shape:', train_set.shape)
#patch_size = 16
batch_size = 2
nb_classes = 3
nb_epoch = 16
#img_rows=132
#img_cols=132

Y_train = np_utils.to_categorical(y_train, nb_classes)
print(Y_train.shape)
train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /=np.max(train_set)

print(train_set.shape)
from keras import regularizers
from sklearn.cross_validation import train_test_split
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, Adadelta
from keras import backend as K
from keras import initializers
from keras.callbacks import Callback
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model = Sequential()
model.add(LSTM(4096,return_sequences=False,input_shape=(16,2048),init='he_normal',kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01),dropout=0.2))
model.add(Dense(1024))
model.add(Dense(nb_classes,activation='softmax'))
sgd=keras.optimizers.SGD(momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
for layer in model.layers:
 print(layer.get_output_at(0).get_shape().as_list())
print(model.summary())
ac1=[]
ac1.append(0)
count=0
model_json = model.to_json()
with open("/home/dgxuser103/Surbhi/9/Surbhi/data/inceptionmodel.json", "w") as json_file:
    json_file.write(model_json)
#model.load_weights(r'/home/dgxuser103/Surbhi/9/Surbhi/data/inceptionmodelweights2.hdf5')
X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, Y_train, test_size=0.1, random_state=4)    
hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),batch_size=batch_size,verbose=1,nb_epoch = nb_epoch,shuffle=True)
model.save_weights('/home/dgxuser103/Surbhi/9/Surbhi/data/inceptionmodelweights.hdf5')
def visualizeHis(hist):
    # visualizing losses and accuracy

    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(nb_epoch)

    loss=plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    print("plotting")
    acc=plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    #plt.show()
    loss.savefig(r'/home/dgxuser103/Surbhi/9/Surbhi/data/inceptionlosssgd16.png')
    acc.savefig(r'/home/dgxuser103/Surbhi/9/Surbhi/data/inceptionaccsgd16.png')

visualizeHis(hist)    
    
    