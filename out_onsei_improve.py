#-*- cording: utf-8 -*-
import numpy as np
#np.random.seed(1337) # for reproducibility
import wave
import pyaudio
from vgg16_like import model_family_cnn
from keras.preprocessing import image
import matplotlib.pyplot as plt
import keras
import time
import cv2
from keras.preprocessing import image
import os

def prediction(imgSrc):
    #np.random.seed(1337) # for reproducibility

    img = np.array(imgSrc)
    img = img.reshape(1, img_rows,img_cols,3)
    img = img.astype('float32')
    img /= 255
    t0=time.time()
    y_pred = model.predict(img)
    return y_pred

num_classes = 2
img_rows,img_cols=128, 128
input_shape = (img_rows,img_cols,3)   #224, 224, 3)
    
model = model_family_cnn(input_shape, num_classes = num_classes)
# load the weights from the last epoch
model.load_weights('params_hirakegoma-61.hdf5', by_name=True) #params_hirakegoma-61.hdf5

print('Model loaded.')
path = "./out_test/figure.jpg"
img_rows,img_cols=128,128
while True:
    #np.random.seed(1337) # for reproducibility
    imgSrc=[]
    if os.path.exists(path)==True:
        
        for i in range(100000):
            i += 1
        
        #img = image.load_img("ohayo1_1_15_out.jpg", target_size=(img_rows,img_cols))
        imgSrc = image.load_img("./out_test/figure.jpg", target_size=(img_rows,img_cols))
        plt.imshow(imgSrc)
        plt.pause(1)
        #plt.show()
        plt.close()
        
        #imgSrc = img  #image.img_to_array(img)
        pred = prediction(imgSrc)
        print(pred[0])

        if pred[0][0]>=0.5:
            filename = "ohayo.wav"
            print("ohayo")
        else:
            filename = "hirakegoma.wav"
            print("hirakegoma")    

        # チャンク数を指定
        CHUNK = 1024
        #filename = "hirakegoma.wav"
        wf = wave.open(filename, "rb")

        # PyAudioのインスタンスを生成
        p = pyaudio.PyAudio()

        # Streamを生成
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

        # データを1度に1024個読み取る
        input = wf.readframes(CHUNK)

        # 実行

        while stream.is_active():    
            output = stream.write(input)
            input = wf.readframes(CHUNK)
            #print(input)
            if input==b'':
                os.remove(path)
                break
        
# ファイルが終わったら終了処理
stream.stop_stream()
stream.close()

p.terminate()

