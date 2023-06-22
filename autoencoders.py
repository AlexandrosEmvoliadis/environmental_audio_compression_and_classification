# -*- coding: utf-8 -*-

"""autoencoders
author: @AlexandrosEmvoliadis
"""

import librosa
import numpy as np
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Input, Concatenate, Lambda, Subtract, UpSampling1D, ZeroPadding1D, Reshape, Flatten, BatchNormalization, Add, ReLU,Softmax
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Flatten, GlobalAveragePooling1D, GlobalMaxPooling1D#, GlobalMeanVarPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Conv1DTranspose, Conv2D, GlobalAveragePooling2D, Lambda, AveragePooling2D, MaxPooling2D
import keras.utils
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend, metrics, callbacks, layers, initializers
import math
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,make_scorer
from sklearn.neighbors import KNeighborsClassifier as knn
from scipy import stats
import soundfile as sf
import os
from keras.applications import ResNet50
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC as svm
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import IsolationForest as iF
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score,f1_score
from operator import add
from sklearn.cluster import DBSCAN,SpectralClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.gaussian_process.kernels import RBF
import random
import functools
import datetime
#import tensorflow_addons as tfa
from pypesq import pesq
from skimage.metrics import structural_similarity as ssim

def load_audio_5_fold(path,p_names,np_names,idx,resample = True):
  """
  Function to load audio from each folder.
  Inputs:
    path    : path of the folder containing the wav files
    p_names : class names related to pollution
    np_names  : class names non-related to pollution
    idx: nindex of folder
    resample: option to resample the wav files. Set to true to test plain ACDNet. Otherwise, test the modified ACDNet (1,22050,1)
  Ouputs:
    train and test sets for both PR/NPR classes
  """

  train_p = []
  test_p = []
  train_np = []
  test_np = []
  for folder in os.listdir(path):
    if folder in np_names:
      category = 1
    elif folder in p_names:
      category = 0
    else:
      #print(folder)
      continue
    for subfolder in os.listdir(path +'/'+ folder):
      print(int(subfolder.split('_')[1]))
      if (int(subfolder.split('_')[1])) == idx-1:
        for file in os.listdir(path + '/' + folder + '/' + subfolder):
          raw,rate = librosa.load(path = path + '/' + folder + '/' + subfolder + '/' + file,sr = 22050,mono = True)
          if resample:
            raw = librosa.resample(raw, orig_sr=22050, target_sr=20150)
          #print(raw.shape)
          if category == 1:
            test_np.append(raw)
          else:
            test_p.append(raw)
      else:
        for file in os.listdir(path + '/' + folder + '/' + subfolder):
          raw,rate = librosa.load(path = path + '/' + folder + '/' + subfolder + '/' + file,sr = 22050,mono = True)
          if resample:
            raw = librosa.resample(raw, orig_sr=22050, target_sr=20150)
          if category == 1:
            train_np.append(raw)
          else:
            train_p.append(raw)
  train_p = [val for vec in train_p for val in vec]
  train_np = [val for vec in train_np for val in vec]
  test_p = [val for vec in test_p for val in vec]
  test_np = [val for vec in test_np for val in vec]
  return np.asarray(train_p),np.asarray(train_np),np.asarray(test_p),np.asarray(test_np)

def process_audio(raw, category,dataset,resample, normalize=0):

    #print(resample)
    """
    Function that preocess audio (AUTH)
    Inputs:
      raw: raw audio waveform
      category: label
      dataset: train or test datasat(deprecated)
    Outputs:
      data: processed raw audio
      truth: labels
    """

    # configuration
    divisor = 0.1
    dBFSThreshold = -96                 # in dBFS
    if resample:
      sample_rate = 20150             # in Hertz
      len = 1.5
    else:
      sample_rate = 22050             # in Hertz
      len = 1
    length = int(sample_rate * len)    # in seconds
    step = int(length*divisor)            # in seconds
    persa_au = 3                        # +/- dB (1 to 3 :: 3)
    persa_snr = 9                       # dB below the mean energy of the sample (6 to 18 :: 10)
    log = False

    # prepare data
    #noise, rate = librosa.load('D:\\PhD\\Datasets\\pink_noise_1h.wav', sr=sample_rate, mono=True)
    #print(raw.shape[0])
    for i in range(0, raw.shape[0]-max(length,step), step):
        if i == 0: count = 0
        column = raw[i:i + length]
        dbFS = 10*np.log10(np.square(column).mean()) if np.square(column).mean() > 0 else -math.inf
        if dbFS > dBFSThreshold:
            count = count + 1

    # load data
    data = np.zeros((count, length))
    truth = np.full(count, category)
    for i in range(0, raw.shape[0]-max(length,step), step):
        if i == 0: count = 0
        column = raw[i:i + length]
        dbFS = 10 * np.log10(np.square(column).mean()) if np.square(column).mean() > 0 else -math.inf

        if dbFS > dBFSThreshold:

            # NONE
            if normalize == 0:
                #
                pass

            # AU
            if normalize == 1:
                #
                column = column * np.random.uniform(pow(10, -persa_au / 20.), pow(10, persa_au / 20.))

            # PERSA
            if normalize == 8:
                column = column[:] / np.sqrt(np.mean(np.square(column)))
                # column = column[:] / np.amax(np.abs(column))
                column = column * np.random.uniform(pow(10, -persa_au/20.), pow(10, persa_au/20.))

            # PERSA+
            if normalize == 9:
                n = noise[i%(noise.shape[0]-length):i%(noise.shape[0]-length)+length]

                p_s = 10 * np.log10(np.square(column).mean() + np.finfo(float).eps)
                p_n = 10 * np.log10(np.square(n).mean() + np.finfo(float).eps)
                n = n * pow(10, (p_s - p_n - persa_snr) / 20.)
                column = column + n
                # print('Signal PW: {:.2f} | Noise PW: {:.2f} | Noise PWC: {:.2f} '.format(p_s, p_n, 10*np.log10(np.square(n).mean())))

                column = column[:] / np.sqrt(np.mean(np.square(column)))
                column = column * np.random.uniform(pow(10, -persa_au / 20.), pow(10, persa_au / 20.))

            if log:
                sign = np.sign(column)
                column = np.log10(np.abs(column) + 1)
                column = np.multiply(column, sign)

            data[count, :] = column
            # plt.plot(column[500:600])
            # plt.show()

            count = count + 1

    print('Class: {}, Full Size: {}, Filtered Size: {}, Window Length: {:.2f}s, Step Length: {:.2f}s'.format(category, int(raw.shape[0]/step), data.shape[0], length/sample_rate, step/sample_rate))

    noise = None
    data = data.reshape(data.shape[0], data.shape[1], 1)

    truth = truth.reshape(truth.shape[0], 1)
    return data, truth

def to_model(np_names,p_names,path,epochs,folds,fold,resample):
  """
  Function that prepares data for the model
  Inputs:
    np_names: names of classes not related to pollution
    p_names: names of classes related to pollution
    path: path of the folder that contains the wav files
    epochs: number of training steps for the classifier
    folds: number of folders
    fold: current folder
  Outputs:
    prepared data for the model

  Un-comment ..._np and comment ..._p to train for the NPR Class
  """
  print("Prepare Data for the Model")
  #load audio, modification of previous version
  x_train_polluting,x_train_nonpolluting,x_test_polluting,x_test_nonpolluting = load_audio_5_fold(path,p_names,np_names,fold,resample)
  x_train_p,y_train_p = process_audio(x_train_polluting,0,'train',resample)
  # x_train_np,y_train_np = process_audio(x_train_nonpolluting,1,'train',resample)
  # process audio (ex load_audio)
  x_test_p,y_test_p = process_audio(x_test_polluting,0,'test',resample)
  # x_test_np,y_test_np = process_audio(x_test_nonpolluting,1,'test',resample)
  # x_train_np = x_train_np.astype('float16')
  # x_test_np = x_test_np.astype('float16')

  x_train_p = x_train_p.astype('float16')
  x_test_p = x_test_p.astype('float16')

  #release memory
  del x_train_nonpolluting,x_test_nonpolluting, x_train_polluting, x_test_polluting

  #shuffling
  # np.random.shuffle(x_train_p)
  # np.random.shuffle(x_train_np)
  # np.random.shuffle(x_test_p)
  # np.random.shuffle(x_test_np)
  #data preparation
  x_train = x_train_p
  y_train = y_train_p

  x_test = x_test_p
  y_test = y_test_p


  # y_train = keras.utils.to_categorical(y_train,2)
  # y_test = keras.utils.to_categorical(y_test,2)

  #release memory
  del x_train_p,x_test_p,y_train_p,y_test_p
  return (x_train,y_train,x_test,y_test)

def PSNR(original,reconstructed):
  m_PSNR = []
  for orig, recon in zip(original,reconstructed):
    M1 = np.abs(np.min(orig))
    M2 = np.abs(np.max(orig))
    max = M2
    if M1 > M2:
      max = M1
    mse = np.sum((orig - recon) ** 2)/22050
    m_PSNR.append(10 * np.log10(max ** 2 / mse))
  return np.mean(m_PSNR)

def SSIM(original,reconstructed):
  m_SSIM = []
  for orig,recon in zip(original,reconstructed):
    ssiM = ssim(orig.flatten(),reconstructed.flatten())
    m_SSIM.append(ssiM)
  return np.mean(m_SSIM)

def PESQ(original,reconstructed):
  m_PESQ = []
  for orig,recon in zip(original,reconstructed):
    psq = pesq(orig,recon)
    m_PESQ.append(psq)
  return np.mean(psq)

def SE_Block(input,ratio):
  init = input
  filters = init.shape[-1]
  print(filters)
  sh = input.shape
  m = GlobalAveragePooling1D()(init)
  print(m.shape)
  m = tf.reshape(m,(tf.shape(input)[0],1,filters))
  print(m.shape)
  m = Dense(filters//ratio,activation = 'relu', use_bias = 'False')(m)
  m = Dropout(0.1)(m)
  print(m.shape)
  m = Dense(filters,activation = 'sigmoid', use_bias = 'False')(m)
  m = Dropout(0.1)(m)
  print(m.shape)
  m = input*m
  print(m.shape)
  return m

def autoencoder(np_names,p_names,path,epochs,folds,fold,resample, use_SE):
  x_train,_,x_test,_ = to_model(np_names,p_names,path,epochs,folds,fold,resample)
  if not os.path.exists('./autoencoders'):
     os.mkdir('./autoencoders')
  # input = Input(x_train.shape[1:])
  # x = model.layers[2](input)
  # for layer in model.layers[1:]:
  #   layer.trainable = True
  #   x = layer(x)
  # autoencoder = Model(inputs = [input],outputs = [x])
  lr = 5e-4
  val_size = 0.1
  length = 5
  step = int(1 / val_size * length)
  for i in range(0, x_train.shape[0] - length, step):
      if i == 0: r = []
      r.extend(range(i, i + length))
  x_val = x_train[r, :]
  x_train = np.delete(x_train, r, axis=0)
  #callback = [callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, restore_best_weights=True)]

  x = Input((22050,1))
  m = Conv1D(kernel_size = 7,filters = 32,kernel_regularizer = l2(1e-4),activation = 'tanh')(x)
  m = AveragePooling1D(pool_size = 2)(m)
  m = Dropout(0.1)(m)

  m = Conv1D(kernel_size = 7,filters = 64,kernel_regularizer = l2(1e-4),activation = 'tanh')(m)
  m = AveragePooling1D(pool_size = 2)(m)
  m = Dropout(0.1)(m)

  m = Conv1D(kernel_size = 7,filters = 96,kernel_regularizer = l2(1e-4),activation = 'tanh')(m)
  m = Dropout(0.1)(m)
  if use_SE:
    m = SE_Block(m,8)
  m = AveragePooling1D(pool_size = 4)(m)
  m = Dropout(0.1)(m)


  BottleNeck = Dense(1,kernel_regularizer = l2(1e-4),activation = 'tanh')(m)

  m = Conv1DTranspose(kernel_size = 8, filters = 96, kernel_regularizer = l2(1e-4),activation = 'tanh',strides = 4,output_padding = 3, padding = 'valid')(BottleNeck)
  m = Dropout(0.1)(m)
  if use_SE:
    m = SE_Block(m,8)

  m = Dropout(0.1)(m)

  m = Conv1DTranspose(kernel_size = 8, filters = 64, kernel_regularizer = l2(1e-4),activation = 'tanh',strides = 2,output_padding = 1,padding = 'valid')(m)
  m = Dropout(0.1)(m)


  m = Conv1DTranspose(kernel_size = 9, filters = 32, kernel_regularizer = l2(1e-4),activation = 'tanh',strides = 2,output_padding = 1,padding = 'valid')(m)
  m = Dropout(0.1)(m)

  m = Conv1D(kernel_size = 1,filters = 1,activation = 'tanh')(m)

  model = Model(inputs = [x], outputs = [m])
  model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(lr), metrics=['mae'])
  print("Start Training!")
  model.fit(x_train, x_train, batch_size=128, epochs=epochs, verbose=2, validation_data=(x_val, x_val))
  
  predicted = model.predict(x_test)
  psnr = PSNR(x_test,predicted)
  ssiM = SSIM(x_test,predicted)

  if use_SE:
      if not os.path.exists('./autoencoders/se_cae'):
         os.mkdir('./autoencoders/se_cae')
      model.save('./autoencoders/se_cae/' + str(fold) + '.h5')
  else:
      if not os.path.exists('./autoencoders/se_cae'):
         os.mkdir('./autoencoders/cae')
      model.save('./autoencoders/cae/' + str(fold) + '.h5')
  return psnr,

if __name__ == '__main__':
  # np_names = ['thunderstorm','cat','dog','sea_waves','chirping_birds','insects','wind','crickets']
  np_names = []
  p_names = ['train','helicopter','car_horn','fireworks','airplane','hand_saw','chainsaw','engine','siren']
  # p_names = []
  path = './classes_5_fold'
  preds = []
  for idx in range(1, 6):  # folds
      if 'acc' not in locals(): acc = np.zeros((5,3))
      # np_names,p_names,path,epochs,folds,fold,resample
      acc[idx-1] = autoencoder(np_names,p_names,path,70,5,idx,resample = False, use_SE = True)
      print('Mean PSNR score   for fold {}/{}: {:.1f} dB'.format(idx, 5, (acc[idx-1][0])))
      print('Mean SSIM score   for fold {}/{}: {:.1f}'.format(idx, 5, (acc[idx-1][1])))
      print('Mean PESQ score   for fold {}/{}: {:.1f}'.format(idx, 5, (acc[idx-1][2])))
  print('Mean PSNR score  total : {:.1f} ±{:.1f} dB'.format(np.mean(acc[:][0]), np.std(acc[idx][0])))
  print('Mean SSIM score  total : {:.1f} ±{:.1f} dB'.format(np.mean(acc[:][1]), np.std(acc[idx][1])))
  print('Mean PESQ score  total : {:.1f} ±{:.1f} dB'.format(np.mean(acc[:][-1]), np.std(acc[idx][-1])))
