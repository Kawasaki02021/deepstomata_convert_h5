import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL
from keras.applications.vgg16 import VGG16

X = []
Y = []


for picture in list_pictures("C:/Users/Rx-80/Downloads/stomata56/stomataval/close"):
	img = img_to_array(load_img(picture, target_size=(56,56)))
	X.append(img)
	Y.append(0)

for picture in list_pictures("C:/Users/Rx-80/Downloads/stomata56/stomataval/open"):
	img = img_to_array(load_img(picture, target_size=(56,56)))
	X.append(img)
	Y.append(1)

for picture in list_pictures("C:/Users/Rx-80/Downloads/stomata56/stomataval/partially open"):
	img = img_to_array(load_img(picture, target_size=(56,56)))
	X.append(img)
	Y.append(2)
for picture in list_pictures("C:/Users/Rx-80/Downloads/stomata56/stomataval/false positive"):
	img = img_to_array(load_img(picture, target_size=(56,56)))
	X.append(img)
	Y.append(3)

X = np.asarray(X)
Y = np.asarray(Y)

X = X.astype('float32')
X = X / 255.0

Y = np_utils.to_categorical(Y, 4)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=111)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from keras.layers import Dense, GlobalAveragePooling2D,Input
import os
n_categories=4
batch_size=100
base_model=VGG16(weights='imagenet',include_top=False,
                 input_tensor=Input(shape=(56,56,3)))
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
prediction=Dense(n_categories,activation='softmax')(x)
model=Model(input=base_model.input,output=prediction)
model.load_weights("C:/test/stomata_classify.h5")
#fix weights before VGG16 14layers
for layer in base_model.layers[:15]:
    layer.trainable=False

model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train, batch_size, nb_epoch=100, validation_data = (X_test, y_test), verbose = 1)                   
saver = tf.train.Saver()
sess= tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
saver.save(sess, "C:/Users/Rx-80/Downloads/model/model.ckpt")
import h5py
eval = model.evaluate(X_test, y_test, verbose=1)
print("正解率=", eval[1], "ロス=", eval[0])
model.save('C:/test/stomata_classify_test00.hdf5')
json_string = model.to_json()
open('C:/test/stomata_classify_test00.json', 'w').write(json_string)
model.load_weights('C:/test/stomata_classify_test00.hdf5')