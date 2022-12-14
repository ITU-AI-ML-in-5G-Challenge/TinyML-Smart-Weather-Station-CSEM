import keras

from keras import regularizers
from keras.layers import Lambda, Input, GlobalAveragePooling1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dense
# from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization, Concatenate, Softmax
# from tensorflow.keras.activations import softmax
from keras.models import Sequential, Model

from keras import layers

from constants_ml import *




class Conv_block(keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, strides, l2_factor, pool_size):
        super(Conv_block, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.l2_factor = l2_factor
        self.pool_size = pool_size
           

    def build(self, input_shape):
        self.conv = Conv1D(self.num_filters,
                                kernel_size=self.kernel_size,
                                strides=self.strides,
                                padding='same',
                                kernel_initializer='glorot_uniform',
                                kernel_regularizer=regularizers.l2(l=self.l2_factor),
                                input_shape=input_shape)
        
        self.batchnorm = BatchNormalization()
        self.activation = Activation('relu')
        self.max_pool = MaxPooling1D(pool_size=self.pool_size, strides=None)


    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x


class M3(tf.keras.Model):
    def __init__(self, audio_length, num_classes, l2_factor):
        super(M3, self).__init__()
        self.in_shape = [audio_length, 1]
        self.conv_block_1 = Conv_block(num_filters=(MAX_RAM_MEMORY/ audio_length) * 2,  
                                        kernel_size=100, 
                                        strides=4, 
                                        l2_factor=l2_factor, 
                                        pool_size=4)

        self.conv_block_2 = Conv_block(num_filters=80, kernel_size=3, strides=1, l2_factor=l2_factor, pool_size=4)
        self.global_avg_pool = Lambda(lambda x: keras.backend.mean(x, axis=1))
        self.dense = Dense(num_classes, activation='softmax')

        # Get output layer with `call` method
        self.input_layer = Input(shape=(audio_length, 1))
        self.out = self.call(self.input_layer)
        # Reinitial
        super(M3, self).__init__(
            inputs=self.input_layer,
            outputs=self.out)


    def call(self, inputs, training=False):
        print(inputs.shape)
        x = self.conv_block_1(inputs, training)
        x = self.conv_block_2(x, training)
        x = self.global_avg_pool(x)
        x = self.dense(x)
        return x


class M5(tf.keras.Model):
    def __init__(self, audio_length, num_classes, l2_factor):
        super(M5, self).__init__()
        self.in_shape = [audio_length, 1]
        
        self.conv_block_1 = Conv_block(num_filters=(MAX_RAM_MEMORY/ audio_length) * 2,  
                                        kernel_size=100, 
                                        strides=4, 
                                        l2_factor=l2_factor, 
                                        pool_size=4, 
                                        input_shape=self.in_shape)

        self.conv_block_2 = Conv_block(num_filters=64, kernel_size=3, strides=1, l2_factor=l2_factor, pool_size=4)                  
        self.conv_block_3 = Conv_block(num_filters=258, kernel_size=3, strides=1, l2_factor=l2_factor, pool_size=4)
        self.conv_block_4 = Conv_block(num_filters=128, kernel_size=3, strides=1, l2_factor=l2_factor, pool_size=4)
        
        self.global_avg_pool = Lambda(lambda x: keras.backend.mean(x, axis=1))
        self.dense = Dense(num_classes, activation='softmax')


    def call(self, inputs):
        x = self.conv_block_1(inputs)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.global_avg_pool(x)
        x = self.dense(x)
        return x



def m3(audio_length, num_classes=10, l2_factor=0.0001):
    print('Using Model M3')
    m = Sequential()
    # 256 -> 16
    m.add(Conv1D(int(MAX_RAM_MEMORY/ audio_length) * 4,
                 input_shape=[audio_length, 1],
                 #TODO : remark : changed from 80 to 1000
                 kernel_size=100,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=l2_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    #64 -> 80
    m.add(Conv1D(80,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=l2_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))

    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Lambda(lambda x: keras.backend.mean(x, axis=1))) # Same as GAP for 1D Conv Layer
    m.add(Dense(num_classes, activation='softmax'))
    return m


##################################################################################################################
##################################################################################################################
##################################################################################################################

def final_model(audio_length, num_classes_rain, num_classes_wind):
    print(audio_length)
    print(num_classes_rain)
    print(num_classes_wind)
    inputs = Input(shape=[audio_length, 1], name="input")
    Feature_extractor = m5_opt(audio_length=audio_length, num_classes=num_classes_rain+num_classes_wind, activation=None)
    Feature_extractor.summary()
    x = Feature_extractor(inputs)
    print(x.shape)
    output_rain = Softmax(name='rain')(x[:,0:num_classes_rain])
    print(output_rain.shape)
    output_wind = Softmax(name='wind')(x[:,num_classes_rain:])
    # return Model(inputs=inputs, outputs=output_wind, name='final_model')
    # outputs = Concatenate()([output_rain, output_wind])
    # print(outputs.shape)
    # return Model(inputs=inputs, outputs=outputs, name='final_model')
    return Model(inputs=inputs, outputs=[output_rain, output_wind], name='final_model')





def m5_opt(audio_length, num_classes=10, activation='softmax'):
    print('Using Model M5')
    l1_factor = 1e-5
    filter_1 = int((MAX_RAM_MEMORY/ audio_length) * 2) #37
    kernel_1 = 70
    stride_1 = 5
    pool_size = 4
    m = Sequential()
    #kernel size 100 d
    m.add(Conv1D(filter_1 ,
                 input_shape=[audio_length, 1],
                 kernel_size=kernel_1,
                 strides=stride_1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l1(l1=l1_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=pool_size, strides=None))
    filter_2 = 96
    kernel_2 = 4
    m.add(Conv1D(filter_2,
                 kernel_size=kernel_2,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l1(l1=l1_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    # m.add(MaxPooling1D(pool_size=4, strides=None))
    filter_3 = 96
    kernel_3 = 3
    m.add(Conv1D(filter_3,
                 kernel_size=kernel_3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l1(l1=l1_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    filter_3 = 96
    kernel_3 = 3
    m.add(Conv1D(filter_3,
                 kernel_size=kernel_3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l1(l1=l1_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=pool_size, strides=None))
    filter_4 = 119
    kernel_4 = 3
    m.add(Conv1D(filter_4,
                 kernel_size=kernel_4,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l1(l1=l1_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=pool_size, strides=None))
    # m.add(Dense(200, activation='relu'))
    m.add(Lambda(lambda x: keras.backend.mean(x, axis=1))) # Same as GAP for 1D Conv Layer
    m.add(Dense(110, activation='relu'))
    m.add(Dense(100, activation='relu'))
    m.add(Dense(num_classes, activation=activation))
    return m



##################################################################################################################
##################################################################################################################
##################################################################################################################



def m5(audio_length, num_classes=10, l2_factor=0.0001):
    print('Using Model M5')
    m = Sequential()
    #kernel size 100
    m.add(Conv1D(int(MAX_RAM_MEMORY/ audio_length) * 2,
                 input_shape=[audio_length, 1],
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=l2_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(100,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=l2_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    # m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Conv1D(128,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=l2_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    m.add(Conv1D(128,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=l2_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(Lambda(lambda x: keras.backend.mean(x, axis=1))) # Same as GAP for 1D Conv Layer

    m.add(Dense(num_classes, activation='softmax'))
    return m



def m5_optuna(audio_length, num_classes, trial, l2_factor=0.0001):
    print('Using Model M5')
    m = Sequential()
    #kernel size 100
    filter_1 = trial.suggest_int("filter_1", int(MAX_RAM_MEMORY/ audio_length), int(MAX_RAM_MEMORY/ audio_length) * 2)
    kernel_1 = trial.suggest_categorical("kernel_1", [50, 60, 70, 80])
    stride_1 = trial.suggest_categorical("stride_1", [4, 5])
    pool_size = trial.suggest_categorical("pool_size", [3, 4, 5])
    l1_factor = trial.suggest_float("l1_factor", 1e-5, 1e-2, log=True)
    l2_factor = trial.suggest_float("l2_factor", 1e-5, 1e-2, log=True)
    use_dense = trial.suggest_int('use_dense', 0, 1)
    dense_num = trial.suggest_int("dense_num", 20, 60)

    m.add(Conv1D(filter_1,
                 input_shape=[audio_length, 1],
                 kernel_size=kernel_1,
                 strides=stride_1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=l2_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=pool_size, strides=None))
    filter_2 = trial.suggest_categorical("filter_2", [60,70,80,90, 100])
    kernel_2 = trial.suggest_categorical("kernel_2", [3, 4])
    m.add(Conv1D(filter_2,
                 kernel_size=kernel_2,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l1_l2(l1=l1_factor, l2=l2_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    # m.add(MaxPooling1D(pool_size=4, strides=None))
    filter_3 = trial.suggest_categorical("filter_3", [70, 80, 90, 100, 110, 128])
    kernel_3 = trial.suggest_categorical("kernel_3", [3, 4])
    m.add(Conv1D(filter_3,
                 kernel_size=kernel_3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l1_l2(l1=l1_factor, l2=l2_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=pool_size, strides=None))

    filter_4 = trial.suggest_categorical("filter_4", [70, 80, 90, 100, 110, 128])
    kernel_4 = trial.suggest_categorical("kernel_4", [3, 4])
    m.add(Conv1D(filter_4,
                 kernel_size=kernel_4,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l1_l2(l1=l1_factor, l2=l2_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=pool_size, strides=None))
    m.add(Lambda(lambda x: keras.backend.mean(x, axis=1))) # Same as GAP for 1D Conv Layer
    if use_dense > 0:
        m.add(Dense(dense_num, activation='relu'))
    m.add(Dense(num_classes, activation='softmax'))
    return m





def m11(audio_length, num_classes=10, l2_factor=0.0001):
    print('Using Model M11')
    m = Sequential()
    m.add(Conv1D(int(MAX_RAM_MEMORY/ audio_length) * 3,
                 input_shape=[audio_length, 1],
                 kernel_size=70,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=l2_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(2):
        m.add(Conv1D(32,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=l2_factor)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(2):
        m.add(Conv1D(32,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=l2_factor)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(3):
        m.add(Conv1D(32,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=l2_factor)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(2):
        m.add(Conv1D(32,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=l2_factor)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))

    m.add(Lambda(lambda x: keras.backend.mean(x, axis=1))) # Same as GAP for 1D Conv Layer
    m.add(Dense(num_classes, activation='softmax'))
    return m






def m_rec(audio_length, num_classes=10, l2_factor=0.0001):
    from keras.layers.recurrent import LSTM
    print('Using Model LSTM 1')
    m = Sequential()
    m.add(Conv1D(int(MAX_RAM_MEMORY/ audio_length) * 2,
                 input_shape=[audio_length, 1],
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=l2_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))
    m.add(LSTM(32,
               kernel_regularizer=regularizers.l2(l=l2_factor),
               return_sequences=True,
               dropout=0.2))
    m.add(LSTM(32,
               kernel_regularizer=regularizers.l2(l=l2_factor),
               return_sequences=False,
               dropout=0.2))
    m.add(Dense(32))
    m.add(Dense(num_classes, activation='softmax'))
    return m



def m18(audio_length, num_classes=10, l2_factor=0.0001):
    print('Using Model M18')
    m = Sequential()
    m.add(Conv1D(int(MAX_RAM_MEMORY/ audio_length) * 4,
                 input_shape=[audio_length, 1],
                 kernel_size=80,
                 strides=4,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=l2_factor)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(4):
        m.add(Conv1D(128,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=l2_factor)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(4):
        m.add(Conv1D(64,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=l2_factor)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(4):
        m.add(Conv1D(64,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=l2_factor)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
    m.add(MaxPooling1D(pool_size=4, strides=None))

    for i in range(4):
        m.add(Conv1D(64,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=l2_factor)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))

    m.add(Lambda(lambda x: keras.backend.mean(x, axis=1))) # Same as GAP for 1D Conv Layer
    m.add(Dense(num_classes, activation='softmax'))
    return m








# For m34 Residual, use RepeatVector. Or tensorflow backend.repeat

def identity_block(input_tensor, kernel_size, filters, stage, block, l2_factor=0.0001):
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=l2_factor),
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=l2_factor),
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)

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



def resnet_34(audio_length, num_classes=10, l2_factor=0.0001):
    inputs = Input(shape=(audio_length, 1))

    x = Conv1D(48,
               kernel_size=80,
               strides=4,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=l2_factor))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(3):
        x = identity_block(x, kernel_size=3, filters=48, stage=1, block=i, l2_factor=l2_factor)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(4):
        x = identity_block(x, kernel_size=3, filters=96, stage=2, block=i, l2_factor=l2_factor)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(6):
        x = identity_block(x, kernel_size=3, filters=192, stage=3, block=i, l2_factor=l2_factor)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(3):
        x = identity_block(x, kernel_size=3, filters=384, stage=4, block=i, l2_factor=l2_factor)

    x = GlobalAveragePooling1D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    m = Model(inputs, x, name='m34')
    
    return m


#######################################################################
# from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9660891
#######################################################################

def residual_block(x, filters, conv_num=3,
    activation="relu"):
    # Shortcut
    s = Conv1D(filters, 1, padding="same")(x)

    for i in range(conv_num - 1):
        x = Conv1D(filters, 3,padding="same")(x)
        x = keras.layers.Activation(activation)(x)

    x = Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


def rain_model(audio_length, num_classes):
    inputs = Input(shape=[audio_length, 1], name="input")
    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)
    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = keras.layers.Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax", name="output")(x)
    return Model(inputs=inputs, outputs=outputs, name='rain_model')


def mfcc_model(input_shape, num_classes):
    return Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        layers.Normalization(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes),
    ])



models_dict = {
    'm3' : m3,
    'm5' : m5_opt,
    'm5_optuna' : m5_optuna,
    'm11' : m11,
    'm18' : m18,
    'm34' : resnet_34,
    'rain_model' : rain_model,
    'mfcc_model' : mfcc_model,
    'final' : final_model
}