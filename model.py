#Input image dimension was (samples, 50, 50, 50, 1), channel last array. Cropped images were read and resampled using SciPy (scipy.ndimage.interpolation.zoom), which were then normalized using the functions described below. 
#All nodules and masses were annotated using three-dimensional boundary boxes (i.e., cubes) with a commercial software package (AVIEW, Coreline Soft, Seoul, Korea). 
#Therefore, after resampling, all voxel sizes were isotropic.

import gc
import random
from random import randint
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, LSTM, GRU, Embedding, Concatenate, Conv1D, AveragePooling3D, GlobalAveragePooling1D, BatchNormalization, TimeDistributed
from keras import optimizers, layers, regularizers
from keras.layers import LeakyReLU, ReLU 
 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import math

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Activation, MaxPooling3D, Cropping3D, Deconvolution3D, ZeroPadding3D
from keras.layers import Dropout, Input, BatchNormalization, UpSampling3D, concatenate, GlobalAveragePooling3D
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta, Adam, Nadam
from matplotlib.pyplot import cm
from keras.models import Model 
import keras 
from keras.callbacks import (ReduceLROnPlateau,CSVLogger,EarlyStopping,ModelCheckpoint)
from keras.utils import multi_gpu_model
from keras.models import load_model
from imgaug import augmenters as iaa
from keras.regularizers import l2

#Data augmentation 

aug90 = iaa.Sometimes(1, iaa.Affine(rotate=(-90, -90),mode="constant",cval=(-1000,-1000)))
aug180 = iaa.Sometimes(1, iaa.Affine(rotate=(-180, -180),mode="constant",cval=(-1000,-1000)))
aug270 = iaa.Sometimes(1, iaa.Affine(rotate=(-270, -270),mode="constant",cval=(-1000,-1000))) 
augFlip1 = iaa.Sometimes(1, iaa.Fliplr(1))
augFlip2 = iaa.Sometimes(1, iaa.Flipud(1))
augCon= iaa.Sometimes(1, iaa.ContrastNormalization(0.5, 0.9))
augBlur = iaa.Sometimes(1, iaa.GaussianBlur(sigma=(0.0, 0.8)))
augSharpen = iaa.Sometimes(1, iaa.Sharpen(alpha=0.1, lightness=0.7)) 

def augmentHK(image):
    w1=image.reshape(image.shape[0], image.shape[1],image.shape[2]) 
    k=random.randrange(1,11)  
    global w2
    if k==1:
        w2=aug90.augment_image(w1) 
    elif k==2:
        w2=aug180.augment_image(w1)
    elif k==3:
        w2=aug270.augment_image(w1)
    elif k==4:
        w2=augFlip1.augment_image(w1)
    elif k==5:
        w2=augFlip2.augment_image(w1) 
    elif k==6:
        w2=augCon.augment_image(w1)
    elif k==7:
        w2=augBlur.augment_image(w1) 
    elif k==8:
        w2=augSharpen.augment_image(w1)
    elif k==9:
        noi=np.random.normal(0,0.02,(50,50,50))
        w2=np.add(w1,noi)        
    else:
        w2=w1 
    w3=w2.reshape(w2.shape[0], w2.shape[1],w2.shape[2], 1)
    w4=np.array([w3])
    return w4
    
#Generator

def generator(features, labels, batch_size):
    return_features = features.copy() 
    return_labels = labels.copy()
    batch_features = np.zeros((batch_size, 50, 50, 50, 1))  
    batch_labels = np.zeros((batch_size, 12)) 
    
    while True:
        for i in range(batch_size):
            index = randint(0, len(return_features)-1)
            random_augmented_image, random_augmented_labels = augmentHK(features[index]), np.array([labels[index]])
            batch_features[i] = random_augmented_image[0] 
            batch_labels[i] = random_augmented_labels[0]
        yield batch_features, batch_labels

#Number of time intervals to make a discrete-time survival model 

breaks=np.concatenate((np.arange(0,1200,300),np.arange(1200,3000,600))) 
n_intervals=len(breaks)-1
timegap = breaks[1:] - breaks[:-1]

#Pixel value normalization

MIN_BOUND = -1200.0
MAX_BOUND = 300.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image 
 
MIN_BOUND1 = -1000.0
MAX_BOUND1 = 500.0
    
def normalize1(image):
    image = (image - MIN_BOUND1) / (MAX_BOUND1 - MIN_BOUND1)
    image[image>1] = 1.
    image[image<0] = 0.
    return image 

#Model with DenseNet architecture

def dense_factor(inputs, kernel1):
    h_1 = BatchNormalization()(inputs)
    h_1 = Conv3D(kernel1, (3,3,3), kernel_initializer='he_uniform',padding='same')(h_1)
    output = ReLU()(h_1)  
    return output
    
def transition(x, kernel2, droprate): 
    weight_decay=1E-4
    x = BatchNormalization(axis=-1,gamma_regularizer=l2(weight_decay),beta_regularizer=l2(weight_decay))(x)
    x = ReLU()(x)
    x = Conv3D(kernel2, (1, 1, 1),
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x) 
    x = Dropout(droprate)(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
    return x
    
def dense_block(inputs, numlayers, kernel1):

    concatenated_inputs = inputs

    for i in range(numlayers):
        x = dense_factor(concatenated_inputs, kernel1)
        concatenated_inputs = concatenate([concatenated_inputs, x], axis=-1)

    return concatenated_inputs
 
def DenseNet(kernel1, kernel2, kernel3, numlayers, droprate, addD): 
    weight_decay=1E-4
    model_input1 = Input((50, 50, 50, 1), name='image')  
    nb_layers = 3  
    x= BatchNormalization()(model_input1)
    x = Conv3D(kernel3, (3, 3, 3),
               kernel_initializer="he_uniform", 
               name="initial_conv3D",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = dense_block(x, numlayers, kernel1)
    x = transition(x, kernel2, droprate)  
    x = dense_block(x, numlayers, kernel1)
    x = transition(x, kernel2, droprate)    
    if addD == 1:
        x = dense_block(x, 1, kernel1)
    x = BatchNormalization(axis=-1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = ReLU()(x)
    x = GlobalAveragePooling3D()(x) 
    f1=Dense(n_intervals, kernel_initializer='zeros', bias_initializer='zeros')(x)
     
    out=Activation('sigmoid')(f1)
    model = Model(inputs = [model_input1], outputs = [out])  
    return model 
    
#in our study, hyperparameters were set as follows: kernel1=32, kernel2=48, kernel3=32, droprate=0.2, numlayers=2, and addD=1
       
model=DenseNet(kernel1, kernel2, kernel3, numlayers, droprate, addD)
model=multi_gpu_model(model, gpus=4) # depends on the number of available GPUs
model.compile(loss=surv_likelihood(n_intervals), optimizer=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004))
early_stopping = EarlyStopping(monitor='val_loss', patience=10) 
csv_logger = CSVLogger('./Dense log%s.csv' %str(k), append=True, separator=';')
checkpointer = ModelCheckpoint(filepath='bestmodel at iter%s.h5' %str(k), verbose=1, save_best_only=True, monitor='val_loss', mode='auto')
history=model.fit_generator(generator(images1,y_train, 40), steps_per_epoch = 240, epochs=50, verbose=1, validation_data=(images2, y_tune), callbacks=[early_stopping,csv_logger,checkpointer])
    
