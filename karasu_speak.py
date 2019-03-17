# -*- coding: utf-8 -*-
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
    #for j in range(0,100000,1):
    #    j += 1        

    imgSrc = image.load_img(path, target_size=(img_rows,img_cols))
    #plt.imshow(imgSrc)
    #plt.pause(1)
    #plt.close()

    pred = prediction(imgSrc,model)
    #print(pred[0])
    if pred[0][0]>=0.5:
        filename = "karasu-miyama_out1.wav"
        print("angry")
    elif pred[0][1]>=0.5:
        #filename = "karasu_kero_out3.wav"
        filename = "karasu-normal_out1.wav"
        print("normal") 
    elif pred[0][2]>=0.5:
        #filename = "karasu_kero_out1.wav"
        filename = "karasu-others_out1.wav" #karasu-hageshii_out.wav
        print("others") 
    return filename
    

num_classes = 3
img_rows,img_cols=128, 128
input_shape = (img_rows,img_cols,3)
    
model = model_family_cnn(input_shape, num_classes = num_classes)
# load the weights from the last epoch
model.load_weights('params_karasu-0angry-1normal-2others.hdf5', by_name=True)
print('Model loaded.')
path = "./out_test/figure.jpg"
img_rows,img_cols=128,128
s=0
while True:
    if os.path.exists(path)==True:
        s += 1
        for j in range(0,50000000,1):
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
        wf = wave.open(filename, "rb")

        # チャンク数を指定
        CHUNK1 = 1024
        #filename = "hirakegoma.wav"
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
                