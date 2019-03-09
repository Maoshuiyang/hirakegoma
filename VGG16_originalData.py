from __future__ import print_function
import numpy as np
np.random.seed(1337) # for reproducibility
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from keras.layers import BatchNormalization
from keras.layers import Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.models import Model
import h5py
from getDataSet import getDataSet
import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import cv2
from keras.preprocessing import image
import matplotlib.pyplot as plt

batch_size = 32  #128
num_classes = 2
epochs = 61  #256
img_rows,img_cols=128,128  #128, 128

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train, x_test, y_test = getDataSet(img_rows,img_cols) #X_train,y_train,X_test,y_test

x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.reshape(len(x_train), img_rows*img_cols,3)
x_test = x_test.reshape(len(x_test), img_rows*img_cols,3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def model_family_cnn(input_shape, num_classes=10):
    input_layer = Input(shape=input_shape)

    # Block 1
    conv1_1 = Conv2D(32, (3, 3),name='conv1_1', activation='relu', padding='same')(input_layer)
    conv1_2 = Conv2D(32, (3, 3),name='conv1_2', activation='relu', padding='same')(conv1_1)
    bn1 = BatchNormalization(axis=3)(conv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
    drop1 = Dropout(0.5)(pool1)
    
    # Block 2
    conv2_1 = Conv2D(64, (3, 3),name='conv2_1', activation='relu', padding='same')(drop1)
    conv2_2 = Conv2D(64, (3, 3),name='conv2_2', activation='relu', padding='same')(conv2_1)
    bn2 = BatchNormalization(axis=3)(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
    drop2 = Dropout(0.5)(pool2)
    
    # Block 3
    conv3_1 = Conv2D(256, (3, 3),name='conv3_1', activation='relu', padding='same')(drop2)
    conv3_2 = Conv2D(256, (3, 3),name='conv3_2', activation='relu', padding='same')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3),name='conv3_3', activation='relu', padding='same')(conv3_2)
    conv3_4 = Conv2D(256, (3, 3),name='conv3_4', activation='relu', padding='same')(conv3_3)
    bn3 = BatchNormalization(axis=3)(conv3_4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)
    drop3 = Dropout(0.5)(pool3)
    
    # Block 4
    conv4_1 = Conv2D(512, (3, 3),name='conv4_1', activation='relu', padding='same')(drop3)
    conv4_2 = Conv2D(512, (3, 3),name='conv4_2', activation='relu', padding='same')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3),name='conv4_3', activation='relu', padding='same')(conv4_2)
    conv4_4 = Conv2D(512, (3, 3),name='conv4_4', activation='relu', padding='same')(conv4_3)
    bn4 = BatchNormalization(axis=3)(conv4_4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)
    drop4 = Dropout(0.5)(pool4)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3),name='conv5_1', activation='relu', padding='same')(drop4)
    conv5_2 = Conv2D(512, (3, 3),name='conv5_2', activation='relu', padding='same')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3),name='conv5_3', activation='relu', padding='same')(conv5_2)
    conv5_4 = Conv2D(512, (3, 3),name='conv5_4', activation='relu', padding='same')(conv5_3)
    bn5 = BatchNormalization(axis=3)(conv5_4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(bn5)
    drop5 = Dropout(0.5)(pool5)
    
    x = Flatten()(drop3)
    x = Dense(20*num_classes)(x)  #64
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

#img_rows,img_cols=224, 224   #300, 300
input_shape = (img_rows,img_cols,3)   #224, 224, 3)

model = model_family_cnn(input_shape, num_classes = num_classes)

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
base_lr = 0.0001 #3e-4
#opt = keras.optimizers.Adam(lr=base_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
#opt = keras.optimizers.Adam(lr=base_lr)

# load the weights from the last epoch
#model.load_weights('params_hi-ra-ke-go-ma-o-ha-yo-032-128.hdf5', by_name=True)

print('Model loaded.')


# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

x_train=x_train.reshape(len(x_train),img_rows,img_cols,3)
x_test=x_test.reshape(len(x_test),img_rows,img_cols,3)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

# save weights every epoch
model.save_weights('params_hirakegoma-61.hdf5')
model.save_weights(
      'params_cifar10model_epoch_{0:03d}.hdf5'.format(epochs), True)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

img = image.load_img("./out_test/figure.jpg", target_size=(img_rows,img_cols))
plt.imshow(img)
plt.show()
img = image.img_to_array(img)
#img = img.reshape(1, img_rows*img_cols,3)
img = img.astype('float32')
img /= 255
#img = imgSrc.reshape(1, img_rows,img_cols,3)
#img=cv2.imread("hirakegoma1_1_15_out.jpg")
t0=time.time()
y_pred = model.predict(img)
print(y_pred)

img = image.load_img("./out_test/figure1.jpg", target_size=(img_rows,img_cols))
plt.imshow(img)
plt.show()
imgSrc = image.img_to_array(img)
img = imgSrc.reshape(1, img_rows,img_cols,3)
#img=cv2.imread("hirakegoma1_1_15_out.jpg")
t0=time.time()
y_pred = model.predict(img)
print(y_pred)

for i in range(30,50,1):
    #hirakegoma1_9001_15_out
    #img = image.load_img("hirakegoma.jpg", target_size=(img_rows,img_cols))
    plt.imshow(x_test[i])
    print(y_test[i])
    plt.pause(1)
    imgSrc = image.img_to_array(x_test[i])
    img = imgSrc.reshape(1, img_rows,img_cols,3)
    #img=cv2.imread("hirakegoma1_1_15_out.jpg")
    t0=time.time()
    y_pred = model.predict(img)
    print(y_pred)
