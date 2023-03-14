"""""
 *  \brief     train_vent.py
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

import sys
import os
os.system("module load cuda/11.2")
os.system("module load cudnn/8.1")
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from dataloader_vent import extract_dataset

from models import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np



batch_size = 16
test_size = 0.33
seed = 25
epochs = 200


num_classes = 4

resample_rate = None
normalize = False




if __name__ == '__main__':
    
    print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    
    args = sys.argv
    if len(args) == 2:
        NAME_MODEL = args[1].lower()
    
    if not os.path.exists(STORE_ML_FOLDER_VENT ):
        os.makedirs(STORE_ML_FOLDER_VENT )
    
    print('data_extraction')
    X, labels, audio_samplerate = extract_dataset(PATH_DATA_VENT, resample_rate, normalize)

    X = [x[:AUDIO_LENGTH_VENT] for x in X]

    X = np.expand_dims(X, axis=-1)

    x_tr, x_te, y_tr, y_te = train_test_split(X, labels, test_size=test_size, random_state=seed)    

    y_tr = to_categorical(y_tr, num_classes=num_classes)
    y_te = to_categorical(y_te, num_classes=num_classes)

    print('x train shape :', x_tr.shape, ', y train shape :', y_tr.shape)
    print('x val shape :', x_te.shape, ', y val shape :', y_te.shape)


    print('Model selected:', NAME_MODEL)
    model = models_dict[NAME_MODEL](X[0].shape[0], num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    check_memory(model, True)

    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=15, min_lr=0.00001, verbose=1)
    tb_callback = tf.keras.callbacks.TensorBoard(STORE_ML_FOLDER_VENT + './logs', update_freq=1)

    model.fit(x=x_tr,
              y=y_tr,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle=True,
              validation_data=(x_te, y_te),
              use_multiprocessing = True,
              callbacks=[reduce_lr, tb_callback])


    tf.keras.models.save_model(model, PATH_MODEL_VENT)




