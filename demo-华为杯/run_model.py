import tensorflow as tf
import os
import sys
from keras import optimizers
from keras.optimizers import adam,SGD
from data_gen import data_flow,load_test_data
from models.resnet50 import ResNet50
from keras.layers import Flatten,Dense,Dropout,BatchNormalization,Activation,GlobalAveragePooling2D
from keras.models import Model,load_model
from keras.callbacks import TensorBoard,Callback,ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from glob import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
#import img_gen
from keras import regularizers
import matplotlib.pyplot as plt

num_classes = 40  #the num of classes which your task should classify
input_size = 224  #the input image size of the model
batch_size = 16
learning_rate = 1e-4
max_epochs = 1

def add_new_last_layer(base_model,num_classes):
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.5,name='dropout1')(x)
    x = Dense(512,activation='relu',kernel_regularizer= regularizers.l2(0.0001),name='fc2')(x)
    x = BatchNormalization(name='bn_fc_01')(x)
    x = Dropout(0.5,name='dropout2')(x)
    x = Dense(num_classes,activation='softmax')(x)
    model = Model(inputs=base_model.input,outputs=x)
    return model

def model_fn():
    # setup model
    base_model = ResNet50(weights="imagenet",
                          include_top=False,
                          pooling=None,
                          input_shape=(input_size,input_size, 3),
                          classes=num_classes)
    for layer in base_model.layers:
        layer.trainable = False
    model = add_new_last_layer(base_model,num_classes)
    model.summary()
    # Adam = adam(lr=learning_rate,clipnorm=0.001)
    model.compile(optimizer="adam",loss = 'categorical_crossentropy',metrics=['accuracy'])
    return model

def train_model(train_data_dir):
     # data flow generator
    train_sequence, validation_sequence = data_flow(train_data_dir,batch_size,num_classes,input_size)    
    model = model_fn()
    history = model.fit_generator(
        train_sequence,
        steps_per_epoch = len(train_sequence),
        epochs = max_epochs,
        verbose = 1,
        validation_data = validation_sequence,
        max_queue_size = 10,
        shuffle=True
    )
    model.save('garbage_model.h5')
    print('training done!')

## eval 
def test_single_h5(h5_weights_path,test_data_local):
    # model
    model = model_fn()
    model.load_weights(h5_weights_path,by_name=True)
    img_names,test_datas,test_labels = load_test_data(test_data_local,input_size)

    prediction_list =[]
    tta_num = 5
    for test_data in test_datas:
         ## tta
        predictions = [0*tta_num]
        for i in range(tta_num):
            x_test= test_data[i]
            x_test = x_test[np.newaxis,:,:,:]
            prediction = model.predict(x_test)[0]
            predictions += prediction
        prediction_list.append(predictions)

    right_count = 0
    error_infos = []
    
    for index,pred in enumerate(prediction_list):
        pred_label = np.argmax(pred,axis=0)
        test_label = test_labels[index]
        if pred_label == test_label:
            right_count += 1
        else:
            error_infos.append('%s,%s,%s' % (img_names[index],test_label,pred_label))

    accuracy = right_count / len(test_labels)
    print('accuracy: %s' % accuracy)
    result_file_name = os.path.join(os.path.dirname(h5_weights_path),"%s_accuracy.txt" % os.path.basename(h5_weights_path))

    with open(result_file_name,'w') as f:
        f.write('#predict error files\n')
        f.write('#####################\n')
        f.write('filename,true_label,pred_label\n')
        f.writelines(line + '\n' for line in error_infos)
        f.write("#####################\n")
        f.write('accuracy:%s\n'%accuracy)
    print(f'result save at {result_file_name}')


def main(mode):
    if mode:
        train_model('./garbage_data/train_data/')
    else:
        test_single_h5('./garbage_model.h5','./garbage_data/test_data/')
    
if __name__ == "__main__":    
    main(True)
