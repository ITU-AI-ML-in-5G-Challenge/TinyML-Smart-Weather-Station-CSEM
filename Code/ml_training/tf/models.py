"""""
 *  \brief     models.py
 *  \author    Jonathan Reymond
 *  \version   1.0
 *  \date      2023-02-14
 *  \pre       None
 *  \copyright (c) 2022 CSEM
 *
 *   CSEM S.A.
 *   Jaquet-Droz 1
 *   CH-2000 Neuchâtel
 *   http://www.csem.ch
 *
 *
 *   THIS PROGRAM IS CONFIDENTIAL AND CANNOT BE DISTRIBUTED
 *   WITHOUT THE CSEM PRIOR WRITTEN AGREEMENT.
 *
 *   CSEM is the owner of this source code and is authorised to use, to modify
 *   and to keep confidential all new modifications of this code.
 *
 """

import keras

from keras import regularizers
from keras.layers import Lambda, Input, GlobalAveragePooling1D
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D

# from keras.layers.recurrent import LSTM
from keras.layers import LSTM

from keras.layers.core import Activation, Dense

from tensorflow.keras.layers import BatchNormalization, Concatenate, Softmax
# from tensorflow.keras.activations import softmax
from keras.models import Sequential, Model

import numpy as np
from keras import layers
from constants_ml import *



def pseudo_Conv1D(x, filters, kernel_size, strides=1, l1_factor=0, l2_factor=0):
    '''Layer that acts as a convolution 1D, but uses internally conv2D for compatibility with QAT

    Args:
        x (input): input
        filters (int): number of filters
        kernel_size (int): kernel size
        strides (int, optional): strides. Defaults to 1.
        l1_factor (int, optional): l1 regulation factor. Defaults to 0.
        l2_factor (int, optional): l2 regulation factor. Defaults to 0.

    Returns:
        output: input passed through convolution
    '''

    x = tf.keras.layers.Conv2D(filters, 
                                kernel_size=(1, kernel_size), 
                                strides=strides,
                                padding='same',
                                kernel_initializer='glorot_uniform',
                                kernel_regularizer=regularizers.L1L2(l1=l1_factor, l2=l2_factor))(x)
    # batchnormalization only works after conv2D with QAT
    return x
    


def conv_block(x, filters, kernel_size, strides=1, l1_factor=0, l2_factor=0):
    '''Convolution block : convolution --> batchnormalization --> relu

    Args:
        x (input): input
        filters (int): number of filters
        kernel_size (int): kernel size
        strides (int, optional): strides number. Defaults to 1.
        l1_factor (int, optional): l1 regulation factor. Defaults to 0.
        l2_factor (int, optional): l2 regulation factor. Defaults to 0.

    Returns:
        output: input passed through convolution block
    '''
    
    # x = Conv1D(filters,
    #             kernel_size=kernel_size,
    #             strides=strides,
    #             padding='same',
    #             kernel_initializer='glorot_uniform',
    #             kernel_regularizer=regularizers.L1L2(l1=l1_factor, l2=l2_factor))(x)
    x = pseudo_Conv1D(x, filters, kernel_size, strides, l1_factor, l2_factor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


##################################################################################################################
##########################################       Audio Models        #############################################
##################################################################################################################

def m3(x, num_outputs=None, activation='softmax'):
    l2_factor=0.0001
    input_dim_flat = np.prod(x.shape[1:])
    
    x = conv_block(x, 
                   filters=int(MAX_RAM_MEMORY/ np.prod(input_dim_flat)) * 4, 
                   kernel_size=100, strides=4, l2_factor=l2_factor)
    x = MaxPooling1D(pool_size=4, strides=None)(x)
    x = conv_block(x, 
                   filters=80, kernel_size=3, strides=1, l2_factor=l2_factor)
    x = MaxPooling1D(pool_size=4, strides=None)(x)
    x = Lambda(lambda x: keras.backend.mean(x, axis=1))(x) #GAP
    if num_outputs:
        x = Dense(num_outputs, activation=activation)(x)
    return x

def m3_rec(x, num_outputs=None, activation='softmax'):
    l2_factor=0.0001
    input_dim_flat = np.prod(x.shape[1:])
    
    x = conv_block(x, 
                   filters=int(MAX_RAM_MEMORY/ np.prod(input_dim_flat)) * 4, 
                   kernel_size=100, strides=4, l2_factor=l2_factor)
    x = MaxPooling1D(pool_size=4, strides=None)(x)
    x = conv_block(x, 
                   filters=80, kernel_size=3, strides=1, l2_factor=l2_factor)
    x = MaxPooling1D(pool_size=4, strides=None)(x)
    x = LSTM(128,
               kernel_regularizer=regularizers.l2(l=l2_factor),
               return_sequences=True)(x)
    x = LSTM(128,
               kernel_regularizer=regularizers.l2(l=l2_factor),
               return_sequences=False)(x)
    # x = Lambda(lambda x: keras.backend.mean(x, axis=1))(x) #GAP
    if num_outputs:
        x = Dense(num_outputs, activation=activation)(x)
    return x
    
# new m5, that works for only for single input for QAT
def m5(x, num_outputs=None, activation='softmax'):
    l1_factor = 1e-5
    input_dim_flat = np.prod(x.shape[1:])
    
    x = conv_block(x, 
                   filters=int(MAX_RAM_MEMORY/ np.prod(input_dim_flat)) * 2, 
                   kernel_size=70, strides=5, l1_factor=l1_factor)
    x = MaxPooling2D(pool_size=(1,4), strides=None)(x)
    x = conv_block(x, 
                   filters=96, kernel_size=4, l1_factor=l1_factor)
    x = conv_block(x, 
                   filters=96, kernel_size=3, l1_factor=l1_factor)
    x = conv_block(x, 
                   filters=96, kernel_size=3, l1_factor=l1_factor)
    x = MaxPooling2D(pool_size=(1,4), strides=None)(x)
    x = conv_block(x, 
                   filters=119, kernel_size=3, l1_factor=l1_factor)
    x = MaxPooling2D(pool_size=(1,4), strides=None)(x)
    
    output_1, output_2 = x.shape[2:]
    x = tf.keras.layers.Reshape((output_1, output_2))(x)
    
    x = Lambda(lambda x: keras.backend.mean(x, axis=1))(x)
    x = Dense(110, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    if num_outputs:
        x = Dense(num_outputs, activation=activation)(x)
    return x

# old m5, that works for multiple outputs
# def m5(x, num_outputs=None, activation='softmax'):
#     l1_factor = 1e-5
#     input_dim_flat = np.prod(x.shape[1:])
    
#     x = conv_block(x, 
#                    filters=int(MAX_RAM_MEMORY/ np.prod(input_dim_flat)) * 2, 
#                    kernel_size=70, strides=5, l1_factor=l1_factor)
#     x = MaxPooling1D(pool_size=4, strides=None)(x)
#     x = conv_block(x, 
#                    filters=96, kernel_size=4, l1_factor=l1_factor)
#     x = conv_block(x, 
#                    filters=96, kernel_size=3, l1_factor=l1_factor)
#     x = conv_block(x, 
#                    filters=96, kernel_size=3, l1_factor=l1_factor)
#     x = MaxPooling1D(pool_size=4, strides=None)(x)
#     x = conv_block(x, 
#                    filters=119, kernel_size=3, l1_factor=l1_factor)
#     x = MaxPooling1D(pool_size=4, strides=None)(x)
    
#     x = Lambda(lambda x: keras.backend.mean(x, axis=1))(x)
#     x = Dense(110, activation='relu')(x)
#     x = Dense(100, activation='relu')(x)
#     if num_outputs:
#         x = Dense(num_outputs, activation=activation)(x)
#     return x


def m11(x, num_outputs=None, activation='softmax'):
    l1_factor = 1e-5
    input_dim_flat = np.prod(x.shape[1:])
    
    x = conv_block(x, 
                   filters=int(MAX_RAM_MEMORY/ np.prod(input_dim_flat)) * 2, 
                   kernel_size=70, strides=4, l1_factor=l1_factor)
    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for _ in range(2):
        x = conv_block(x, 
                       filters=32, kernel_size=3, l1_factor=l1_factor)
        x = conv_block(x, 
                       filters=32, kernel_size=3, l1_factor=l1_factor)
        x = MaxPooling1D(pool_size=4, strides=None)(x)
        
    for _ in range(3):
        x = conv_block(x, 
                       filters=32, kernel_size=3, l1_factor=l1_factor)
    x = MaxPooling1D(pool_size=4, strides=None)(x)
    
    for _ in range(3):
        x = conv_block(x, 
                       filters=32, kernel_size=3, l1_factor=l1_factor)
        
    x = Lambda(lambda x: keras.backend.mean(x, axis=1))(x) #GAP
    if num_outputs:
        x = Dense(num_outputs, activation=activation)(x)
    return x


def m18(x, num_outputs=None, activation='softmax'):
    l2_factor = 1e-4
    input_dim_flat = np.prod(x.shape[1:])
    
    x = conv_block(x, 
                   filters=int(MAX_RAM_MEMORY/ np.prod(input_dim_flat)) * 2, 
                   kernel_size=70, strides=4, l2_factor=l2_factor)
    x = MaxPooling1D(pool_size=4, strides=None)(x)
    
    filters_list = [128, 64, 64, 64]
    for idx, filters in enumerate(filters_list):
        for _ in range(4):
            x = conv_block(x, 
                           filters=filters, kernel_size=3, l2_factor=l2_factor)
        if idx != len(filters_list):
            x = MaxPooling1D(pool_size=4, strides=None)(x)


    x = Lambda(lambda x: keras.backend.mean(x, axis=1))(x) #GAP
    if num_outputs:
        x = Dense(num_outputs, activation=activation)(x)
    return x


def resnet_34(x,  num_outpus=None, activation='softmax'):
    l2_factor=0.0001
    input_dim_flat = np.prod(x.shape[1:])
    
    def identity_block(input_tensor, kernel_size, filters):
        x = conv_block(x, 
                    filters=filters, kernel_size=kernel_size, l2_factor=l2_factor)
        x = Conv1D(filters,
                kernel_size=kernel_size,
                strides=1,
                padding='same',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(l=l2_factor))(x)
        x = BatchNormalization()(x)
        # up-sample from the activation maps.
        # otherwise it's a mismatch. Recommendation of the authors.
        # here we x2 the number of filters.
        # See that as duplicating everything and concatenate them.
        if input_tensor.shape[2] != x.shape[2]:
            x = layers.add([x, Lambda(lambda y: keras.backend.repeat_elements(y, rep=2, axis=2))(input_tensor)])
        else:
            x = layers.add([x, input_tensor])

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    x = conv_block(x, 
                   filters=int(MAX_RAM_MEMORY/ np.prod(input_dim_flat)) * 2, 
                   kernel_size=70, strides=4, l2_factor=l2_factor)
    x = MaxPooling1D(pool_size=4, strides=None)(x)

    filters_l = [48, 96, 192, 384]
    num_repeats = [3, 4, 6, 3]
    assert len(filters_l) == len(num_repeats)
    
    for step in range(len(filters_l)):
        for _ in range(num_repeats[step]):
             x = identity_block(x, kernel_size=3, filters=filters_l[step])
        if step != len(filters_l) - 1:
            x = MaxPooling1D(pool_size=4, strides=None)(x)

    x = GlobalAveragePooling1D()(x)
    if num_outpus:
        x = Dense(num_outpus, activation=activation)(x)
    return x


def m_rec(x, num_outputs=None, activation='softmax'):
    l1_factor = 1e-5
    input_dim_flat = np.prod(x.shape[1:])

    x = conv_block(x, 
                   filters=int(MAX_RAM_MEMORY/ input_dim_flat) * 2, 
                   kernel_size=70, strides=5, l1_factor=l1_factor)
    x = MaxPooling1D(pool_size=4, strides=None)(x)
    x = conv_block(x, 
                   filters=96, kernel_size=4, l1_factor=l1_factor)
    x = MaxPooling1D(pool_size=4, strides=None)(x)
    
    l2_factor = 1e-4
    x = LSTM(128,
               kernel_regularizer=regularizers.l2(l=l2_factor),
               return_sequences=True)(x)
    x = LSTM(128,
               kernel_regularizer=regularizers.l2(l=l2_factor),
               return_sequences=False)(x)
    # if old version of tf where LSTM does not support CuDDNN
    # x = tf.compat.v1.keras.layers.CuDNNLSTM(128,
    #            kernel_regularizer=regularizers.l2(l=l2_factor),
    #            return_sequences=True)(x)
    # x = tf.compat.v1.keras.layers.CuDNNLSTM(128,
    #            kernel_regularizer=regularizers.l2(l=l2_factor),
    #            return_sequences=False)(x) 
    x = Dense(32)(x)
    if num_outputs:
        x = Dense(num_outputs, activation=activation)(x)
    return x


# from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9660891
def rain_model(x, num_outputs=None, activation='softmax'):

    def residual_block(x, filters, conv_num=3, activation="relu"):
        # Shortcut
        s = Conv1D(filters, 1, padding="same")(x)

        for i in range(conv_num - 1):
            x = Conv1D(filters, 3,padding="same")(x)
            x = keras.layers.Activation(activation)(x)

        x = Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Add()([x, s])
        x = keras.layers.Activation(activation)(x)
        return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    
    x = residual_block(x, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)
    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = keras.layers.Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    if num_outputs:
        x = Dense(num_outputs, activation=activation)(x)
    return x

    
    
def get_classifier(x, num_classes):
    '''get output classifier that depends on single or multiple ouputs

    Args:
        x (input): input
        num_classes (dict): dict number of classes for each label studied

    Returns:
        output: output of model
    '''
    if len(num_classes) == 2:
        x = Dense(num_classes['rain'] + num_classes['wind'], activation=None)(x)
        
        
        x_rain = x[:, 0, 0 : num_classes['rain']]
        x_wind = x[:, 0, num_classes['rain'] :]
        

        output_rain = Softmax(name='rain')(x_rain)
        output_wind = Softmax(name='wind')(x_wind)
        outputs=[output_rain, output_wind]
        
    else :
        name, num_outputs = list(num_classes.items())[0]
        x = Dense(num_outputs, activation=None)(x)
        outputs = [Softmax(name=name)(x)]
        
    return outputs
    


##################################################################################################################
##########################################       Sensor Models         ###########################################
##################################################################################################################


def s1(x, num_outputs, activation=None):
    l2_factor = 0.00001
    x = LSTM(128,
               kernel_regularizer=regularizers.l2(l=l2_factor),
               return_sequences=True)(x)
    x = LSTM(64,
               kernel_regularizer=regularizers.l2(l=l2_factor),
               return_sequences=False)(x)
    x = Dense(64)(x)
    if num_outputs:
        x = Dense(num_outputs, activation=activation)(x)
    return x



##################################################################################################################
##################################################################################################################
##################################################################################################################



    

def get_model(num_classes, type_inputs, name_audio_model, name_sensor_model, audio_length=None, sensor_shape=None, num_outputs=None): 
    '''get the ml model

    Args:
        num_classes (dict): dictionnary with the number of classes for each label studied
        type_inputs (enum Input): type input
        name_audio_model (str): name of audio model selected
        name_sensor_model (str): name of sensor model selected
        audio_length (int, optional): audio length of each sample. Defaults to None.
        sensor_shape (tuple, optional): shape of sensor. Defaults to None.
        num_outputs (int, optional): number of nodes for each output of audio/sensor model before the classification layer. Defaults to None.

    Returns:
        keras model: keras model
    '''
    inputs = []
    temp_outputs = []

    if 'audio' in type_inputs.value:
        #TODO, rechange
        audio_shape = [1, audio_length, 1]
        inputs_audio = Input(shape=audio_shape, name="audio")
        inputs.append(inputs_audio)
        
        x_audio = models_dict[name_audio_model](x=inputs_audio, num_outputs=num_outputs, activation=None)
        temp_outputs.append(x_audio)
        
    if 'sensor' in type_inputs.value:
        inputs_sensor = Input(shape=sensor_shape, name="sensor")
        inputs.append(inputs_sensor)
        x_sensor = models_dict[name_sensor_model](x=inputs_sensor, num_outputs=num_outputs, activation=None)
        temp_outputs.append(x_sensor)
    
    if type_inputs is Inputs.AUDIO_SENSOR:
        assert len(inputs) == 2
        # assert num_outputs is not None
        temp_outputs = Concatenate()(temp_outputs)
        # here : add if needed other layers
        temp_outputs =  Dense(64)(temp_outputs)
        temp_outputs = [temp_outputs]
    
    outputs = get_classifier(temp_outputs[0], num_classes)
    
    return Model(inputs=inputs, outputs=outputs, name='final_model')
    



##################################################################################################################
##################################              Other models          ############################################
##################################################################################################################


def m5_optuna(trial, x, input_shape, num_classes, activation='softmax'):
    pool_size = trial.suggest_categorical("pool_size", [3, 4, 5])
    l1_factor = trial.suggest_float("l1_factor", 1e-5, 1e-2, log=True)
    l2_factor = trial.suggest_float("l2_factor", 1e-5, 1e-2, log=True)
    input_sum = np.prod(input_shape)
    
    x = conv_block(x, 
                   filters=trial.suggest_int("filter_1", int(MAX_RAM_MEMORY/ input_sum), int(MAX_RAM_MEMORY/ input_sum) * 2), 
                   kernel_size=trial.suggest_categorical("kernel_1", [50, 60, 70, 80]), 
                   strides=trial.suggest_categorical("stride_1", [4, 5]), 
                   l1_factor=l1_factor, l2_factor=l2_factor)
    x = MaxPooling1D(pool_size=pool_size, strides=None)(x)
    x = conv_block(x, 
                   filters=96, kernel_size=4, l1_factor=l1_factor)

    x = conv_block(x, 
                   filters=trial.suggest_categorical("filter_2", [60,70,80,90, 100]), 
                   kernel_size=trial.suggest_categorical("kernel_2", [3, 4]), 
                   l1_factor=l1_factor, l2_factor=l2_factor)

    x = conv_block(x, 
                   filters=trial.suggest_categorical("filter_3", [70, 80, 90, 100, 110, 128]), 
                   kernel_size=trial.suggest_categorical("kernel_3", [3, 4]), 
                   l1_factor=l1_factor, l2_factor=l2_factor)
    x = MaxPooling1D(pool_size=pool_size, strides=None)(x)

    x = conv_block(x, 
                   filters=trial.suggest_categorical("filter_4", [70, 80, 90, 100, 110, 128]), 
                   kernel_size=trial.suggest_categorical("kernel_4", [3, 4]),
                   l1_factor=l1_factor, l2_factor=l2_factor)
    x = MaxPooling1D(pool_size=pool_size, strides=None)(x)
    x =Lambda(lambda x: keras.backend.mean(x, axis=1))(x)

    use_dense = trial.suggest_int('use_dense', 0, 1)
    dense_num = trial.suggest_int("dense_num", 20, 60)
    if use_dense > 0:
        x = Dense(dense_num, activation='relu')(x)
    x = Dense(num_classes, activation=activation)
    return x


def mfcc_model(x, num_classes=None, activation='softmax'):
    x = layers.Resizing(32, 32)(x)
    x = layers.Normalization()(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    if num_classes:
        x = layers.Dense(num_classes, activation=activation)(x)
    return x




##################################################################################################################

models_dict = {
    'm3' : m3,
    'm3_rec': m3_rec,
    'm5' : m5,
    'm11' : m11,
    'm18' : m18,
    'm34' : resnet_34,
    'm3_rec' : m3_rec,
    'm_rec' : m_rec,
    
    'rain_model' : rain_model,
    
    's1' : s1,
    
    'mfcc_model' : mfcc_model,
    'm5_optuna' : m5_optuna,
    
}
