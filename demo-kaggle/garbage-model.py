#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout
from tensorflow.keras.models  import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import random,os,glob
import matplotlib.pyplot as plt

print(tf.__version__)

def load_data():
    dir_path = './garbage classification/Garbage classification'
    img_list = glob.glob(os.path.join(dir_path, '*/*.jpg'))
    print(len(img_list))
    #print(os.listdir(dir_path))
    train=ImageDataGenerator(horizontal_flip=True, vertical_flip=True,validation_split=0.1,rescale=1./255,
                         shear_range = 0.1,zoom_range = 0.1,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,)
    test=ImageDataGenerator(rescale=1/255,validation_split=0.1)

    train_generator=train.flow_from_directory(dir_path,target_size=(300,300),batch_size=32,
                                          class_mode='categorical',subset='training')
    test_generator=test.flow_from_directory(dir_path,target_size=(300,300),batch_size=32,
                                        class_mode='categorical',subset='validation')

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    print(labels)

    return train_generator,test_generator

def cnn_model():
    model=Sequential()
    model.add(Conv2D(32,(3,3), padding='same',input_shape=(300,300,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) 
    model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) 
    model.add(Conv2D(64,(3,3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) 
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(6,activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model    

def train_model(model,nb_epoch):
    # Train Model
    train_generator,test_generator=load_data()
    checkpoint = ModelCheckpoint("weights-improvement-{epoch:02d}-{val_acc:.2f}.h5", monitor='val_accuracy', verbose=1, save_best_only=False, mode='auto',period=1)
    callbacks_list = [checkpoint]
    
    step_size_train=train_generator.n//train_generator.batch_size #2276//32
    step_size_test =test_generator.n//test_generator.batch_size #251//32
    '''
    history =  model.fit_generator(train_generator,epochs=nb_epoch,steps_per_epoch=step_size_train,
               validation_data=test_generator,validation_steps=step_size_test,callbacks=callbacks_list)
    '''
    history = model.fit(train_generator, epochs=nb_epoch,callbacks=callbacks_list)
    
    model.save("garbage_model.h5")
    # Evaluate Model
    score = model.evaluate(test_generator)
    print(score)
    
def visualizeHis(hist):
    # visualizing losses and accuracy
    train_loss=hist.history['loss']
    train_acc=hist.history['accuracy']
    xc=range(nb_epoch)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss')
    plt.grid(True)

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc')
    plt.grid(True)
    plt.show()

def train():
    start = time.time()

    nb_epoch = 50
    model = cnn_model()
    train_model(model,nb_epoch)
    #visualizeHis(history)

    end = time.time()
    print(end-start)


# 根据预测结果显示对应的文字label
classes_types = ['cardboard', 'glass', 'trash']
#classes_types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
def generate_result(result):
    for i in range(3):
        if(result[0][i] == 1):
            print(i)
            return classes_types[i]
def show(img_path, results):
    # 对结果进行显示
    frame = cv2.imread(img_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, generate_result(results), (10, 140), font, 3, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('img', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict(img_path):
    img = load_img(img_path,target_size=(300,300))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    model = cnn_model()
    model.load_weight("weight.h5")
    results = model.predict(x)
    print(result[0])
    show(img_path,results)

if __name__ == '__main__':
    train()
    #predict()
    
