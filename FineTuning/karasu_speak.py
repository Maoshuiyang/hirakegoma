# -*- coding: utf-8 -*-
from __future__ import print_function
import cv2
import pyaudio
import sys
import time
import wave
import pydub
from pydub import AudioSegment
import moviepy.editor as mp
import datetime
import os
from vgg16_like import model_family_cnn
from keras.preprocessing import image
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras.layers import Input, Reshape, Embedding
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Reshape, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam, SGD

def prediction(imgSrc,model):
    #np.random.seed(1337) # for reproducibility
    img_rows,img_cols=128, 128
    img = np.array(imgSrc)
    img = img.reshape(1, img_rows,img_cols,3)
    img = img.astype('float32')
    img /= 255
    t0=time.time()
    y_pred = model.predict(img)
    return y_pred

def karasu_responder(model,path,img_rows,img_cols):
    imgSrc=[]
    imgSrc = image.load_img(path, target_size=(img_rows,img_cols))
    #plt.imshow(imgSrc)
    #plt.pause(1)
    #plt.close()

    pred = prediction(imgSrc,model)
    print(pred[0])
    if pred[0][0]>=0.5:
        filename = "karasu-miyama_out1.wav"
        print("angry")
    elif pred[0][1]>=0.5:
        filename = "karasu-normal_out1.wav"
        print("normal") 
    else:  # pred[0][2]>=0.5:
        filename = "karasu-others_out1.wav"
        print("others") 
    return filename
    
num_classes = 3
img_rows,img_cols=128, 128
input_shape = (img_rows,img_cols,3)
    
#model = model_family_cnn(input_shape, num_classes = num_classes)

# VGG16モデルと学習済み重みをロード
# Fully-connected層（FC）はいらないのでinclude_top=False）
input_tensor = Input(shape=input_shape)   #x_train.shape[1:]) 
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

###以下追加
from keras.models import Model
layer_name1 = 'block3_pool'
intermediate_layer_model = Model(inputs=vgg16.input,
                                 outputs=vgg16.get_layer(layer_name1).output)
###ここまで

# FC層を構築
top_model = Sequential()
top_model.add(Flatten(input_shape=intermediate_layer_model.output_shape[1:]))  #vgg16からintermediate_layer_modelに変更 
top_model.add(Dense(15*num_classes, activation='relu'))  #256 #20*num_classes
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

# VGG16とFCを接続
#model = Model(input=vgg16.input, output=top_model(vgg16.output))
model = Model(input=vgg16.input, output=top_model(intermediate_layer_model.output))

# load the weights
model.load_weights('params_model_epoch_karasu_50_008block3_pool.hdf5', by_name=True)  #params_model_epoch_karasu_2000.hdf5
print('Model loaded.')
# モデルのサマリを表示
model.summary()
path = "./out_test/figure.jpg"

s=0
while True:
    if os.path.exists(path)==True:
        s += 1
        for j in range(0,10000000,1):
            j += 1        
        """
        if s%3 == 0:
            path="./out_test/figure_angry.jpg"
        elif s%3 == 1:
            path="./out_test/figure_normal.jpg"
        else:
            path="./out_test/figure_others.jpg"
        """    
        filename=karasu_responder(model,path,img_rows,img_cols)
        
        # チャンク数を指定
        CHUNK1 = 1024
        wf = wave.open(filename, "rb")
        
        # PyAudioのインスタンスを生成
        p1 = pyaudio.PyAudio()
        
        # Streamを生成
        stream1 = p1.open(format=p1.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),        
                output=True)
        # データを1度に1024個読み取る
        input1 = wf.readframes(CHUNK1)
        # 実行
        while stream1.is_active():    
            output = stream1.write(input1)
            input1 = wf.readframes(CHUNK1)
            if input1==b'':
                os.remove(path)
                break
                