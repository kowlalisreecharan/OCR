
# coding: utf-8

# In[8]:

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# In[9]:

batch_size = 128
num_classes = 10
epochs = 12


# In[10]:

img_rows, img_cols = 28, 28


# In[39]:

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[41]:

from matplotlib import pyplot as plt
plt.imshow(x_train[0], interpolation='nearest',cmap = 'gray')
plt.show()


# In[43]:

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


# In[50]:

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# In[52]:

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[56]:

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# In[57]:

model.summary()


# In[58]:

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


# In[59]:

model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))


# In[60]:

score = model.evaluate(x_test, y_test, verbose=0)


# In[61]:

model.predict()


# In[112]:

import numpy as np
import cv2

for x in range(10):
    name = 'C:/Users/sricharan/Desktop/pthon_code/digits/'+str(x)+'.png'
    img = cv2.imread(name,0)
    img = (255-img)
    img = cv2.resize(img, (28, 28),interpolation = cv2.INTER_AREA)
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.show()
    img = img.reshape(1, img_rows, img_cols, 1)
    img = img.astype('float32')
    img /= 255
    print("___________________",model.predict_classes(img)[0],"_____________________")


# In[5]:

import cv2
from matplotlib import pyplot as plt
name = 'C:/Users/sricharan/Desktop/pthon_code/digits/'+'9'+'.png'
img = cv2.imread(name,0)
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)
img = cv2.resize(img, (28, 28),interpolation = cv2.INTER_AREA)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.show()


# In[ ]:



