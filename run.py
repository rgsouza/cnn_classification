# -*- coding: utf-8 -*-
#
# Last modification: 14 Apr. 2019
# Author: Rayanne Souza

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as k
from keras.preprocessing.image import img_to_array
from keras.callbacks import EarlyStopping
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


import os, sys, random
import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import re



def load_images(pathName, dim_1, dim_2):
  
  data = []
  labels = []
  
  # Shuffling images
  images = list(paths.list_images(pathName))
  random.seed(30)
  random.shuffle(images)
  
  # Gets image labels
  for imagePath in images:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (dim_1, dim_2))
    image = img_to_array(image)
    data.append(image)
 
    label = imagePath.split(os.path.sep)[-1]
    label = os.path.splitext(label)[0]
    label = re.sub(r'[0-9]+', '', label)
    
    if label.find("_") != -1:
      label = label.split("_")[1]
      
    if label.find("(") != -1: 
      label = re.sub(r'\(.*\)', '', label)
   
    labels.append(label)
    
  
  return data, labels

def CNN(nclass):
  
  img_shape = Input(shape=(256, 256 , 3))
  
  # Block 1
  x = Conv2D(32, kernel_size=3, activation='relu', padding='same')(img_shape)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = Dropout(0.5)(x)
  
  # Block 2
  x = Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = Dropout(0.5)(x)
  
  # Block 3
  x = Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = Dropout(0.5)(x)
  
  # Block 4
  x = Conv2D(512, kernel_size=3, activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = Conv2D(512, kernel_size=3, activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = Dropout(0.5)(x)
  
  # Block 5
  x = Flatten()(x)
  x = Dense(64, activation='relu')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.5)(x)
  x = Dense(nclass, activation='softmax')(x)
  
  return Model(img_shape, x)

if __name__=='__main__':
            
  print("Loading image")
     
  data, labels = load_images("uc_train_256_data/", 256, 256)
  
  
  x_new, y_real = load_images("uc_test_256/", 256, 256)
  print("[info]: Dataset uploaded")
  
  # Normalization between 0 a 1 
  data = np.array(data, dtype="float32") / 255.0
  x_new = np.array(x_new, dtype="float32") / 255.0
  
  # One hot encode 
  labels = np.array(labels)
  lb = LabelBinarizer()
  labels = lb.fit_transform(labels)

 
  # Split the dataset into training and validation set of 80:20 ratio 
  (x_train, x_test, y_train, y_test) = train_test_split(data,
	labels, test_size=0.2, random_state=42)
  
  # Hyperparameters 
  lr = 1.5e-3
  batch_size = 32
  epochs = 50
  opt = Adam(lr=lr)
  
  nClass = np.shape(y_train)[1]
    
 
  # Creates the convolutional network
  model = CNN(nClass)
  model.summary()
  model.compile(loss = 'categorical_crossentropy', optimizer=opt , metrics=['accuracy'])
 

  # Early stop callback for 10 epochs
  earlyStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
  
 
  History = model.fit(x_train, y_train,
          batch_size = batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[earlyStop]
          )
  
  test_score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', test_score[0])
  print('Test accuracy:', test_score[1])
 
  # Shows the train history
  plt.figure(1)
  plt.plot(History.history['loss'], label='train')
  plt.plot(History.history['val_loss'], label='test')
  plt.xlabel('epoch')
  plt.ylabel('Loss')
  plt.legend()
 
  plt.figure(2)
  plt.plot(History.history['acc'], label='train')
  plt.plot(History.history['val_acc'], label='test')
  plt.xlabel('epoch')
  plt.ylabel('Accuracy')
  plt.savefig("Acc_loss.png")
  plt.legend()
  plt.show()
  
  # Predicts on test set
  y_pred = model.predict(x_new)
  idx = np.argmax(y_pred, axis=1)
 
  n_errors = 0
  n_hits = 0
  for i in range(np.size(y_real)):
   
    if y_real[i] != lb.classes_[idx[i]]:
      n_errors += 1
    else:
      n_hits += 1
      
  e_rate = n_errors/(n_errors + n_hits)
  h_rate = n_hits/(n_errors + n_hits)
 
  
  print("Number of images classified: ", (n_errors+n_hits))
  print("Error rate of ", e_rate*100, "%")
  print("Hit rate of ", h_rate*100, "%")



