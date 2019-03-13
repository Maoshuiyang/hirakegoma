import shutil
import os
import datetime

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

os.makedirs('./dog-cat/3karasu/wav/angry', exist_ok=True)
os.makedirs('./dog-cat/3karasu/wav/normal', exist_ok=True)
os.makedirs('./dog-cat/3karasu/wav/others', exist_ok=True)

num_classes = 3
img_rows,img_cols=128, 128
input_shape = (img_rows,img_cols,3)   #224, 224, 3)
    
model = model_family_cnn(input_shape, num_classes = num_classes)
# load the weights from the last epoch
model.load_weights('params_karasu-0angry-1normal-2others.hdf5', by_name=True) #params_hirakegoma-61.hdf5 #params_hirakegoma-0cat-1dog-2pom #params_hirakegoma-0cat-1dog #params_hirakegoma-0cat-1dog-2pom-3karasu

print('Model loaded.')
path = "./out_test/figure.jpg"
path2 = "./dog-cat/3karasu/wav/figure"
img_rows,img_cols=128,128
s=501
    
for i in range(501,1000,1):
    imgSrc=[]
    for j in range(0,100000,1):
        j += 1        
    #img = image.load_img("ohayo1_1_15_out.jpg", target_size=(img_rows,img_cols))
    imgSrc = image.load_img(path2+str(s)+'.jpg', target_size=(img_rows,img_cols))
    plt.imshow(imgSrc)
    plt.pause(1)
    #plt.show()
    plt.close()

    #imgSrc = img  #image.img_to_array(img)
    pred = prediction(imgSrc)
    print(pred[0])
    #filename = path2+str(s)+".wav"
    if pred[0][0]>=0.5:
        filename = path2+"karasu-hageshii_out.wav"
        print("angry") #cat
        #filename2='angry'+str(s)+'.wav'
        path = 'angry'
    elif pred[0][1]>=0.5:
        filename = path2+"karasu_kero_out3.wav"
        print("normal") #dog
        #filename2='normal'+str(s)+'.wav'
        path = 'normal'
    elif pred[0][2]>=0.5:
        filename = path2+"karasu_kero_out1.wav"
        print("others") #pomeranian
        #filename2='others'+str(s)+'.wav'
        path = 'others'
            
    
    dt_now = datetime.datetime.now()
    with open('./dog-cat/3karasu/wav/file.txt', 'a') as f: #w
        f.write(str(dt_now)+'_'+str(s)+': '+path+'  '+str(pred[0])+'\n')
    
    #print(os.listdir('./dog-cat/3karasu/wav'))
    # ['file.txt', 'dir']

    #print(os.listdir('./dog-cat/3karasu/wav/angry'))
    # []
    
    new_path = shutil.move('./dog-cat/3karasu/wav/' + str(s)+'.wav', './dog-cat/3karasu/wav/'+ path)
    new_path = shutil.move('./dog-cat/3karasu/wav/figure' + str(s)+ '.jpg', './dog-cat/3karasu/wav/' + path)

    #print(new_path)
    # temp/dir2/file.txt

    #print(os.listdir('./dog-cat/3karasu/wav'))
    # ['dir']

    #print(os.listdir('./dog-cat/3karasu/wav/angry'))
    # ['file.txt']
    s += 1
        