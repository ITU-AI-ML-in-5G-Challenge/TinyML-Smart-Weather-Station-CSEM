"""""
 *  \brief     Python script used for ML training
 *  \author    Jonathan Reymond, Robin Berguerand, Jona Beysens
 *  \version   1.0
 *  \date      2022-11-14
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import itertools
import joblib

os.system("module load cuda/11.2")
os.system("module load cudnn/8.1")
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau

from models import *
from utils import *

sys.path.insert(0, 'ml_training/')
sys.path.insert(0, 'ml_training/preprocess_data')
from preprocess_data.prepare_data import get_rain_dataset
from preprocess_data.constants import SAMPLING_FREQUENCY, DATE_RECORDING
from constants_ml import *

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from collections import Counter

import tensorflow as tf
import numpy as np
import pandas as pd

import optuna
from optuna.trial import TrialState
import warnings



smote_resampling = True
class_reweighting = True


batch_size = 24
test_size = 0.17
validation_size = 0.17
seed = 24
epochs = 40




def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-2, log=True
        )
        kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    print('------ optimizer -------')
    print(optimizer_selected)
    print(kwargs)
    return optimizer


def objective(trial):
    sys.path.insert(1, 'ml_training/')

    tf.config.run_functions_eagerly(False)
    
    print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
    
    X, labels = get_rain_dataset(MERGE_BUCKET_SIZE, LOWPASS_FREQ, RESAMPLE_FREQUENCY, TO_MFCC, INDEX_HOURS, compress_labels, SPLIT_FACTOR)

    print('Raw dataset label distribution : %s' % Counter(labels))
    
    if smote_resampling:
        # define pipeline
        X, labels = smote_resampler(X, labels, 1.5, 1, seed)
        
    X, labels, num_classes = reformat_dataset(X, labels) 

    x_tr, x_te, y_tr, y_te = train_test_split(X, labels, test_size=test_size + validation_size, random_state=seed)
    x_val, x_te, y_val, y_te = train_test_split(x_te, y_te, test_size=test_size/(test_size + validation_size), random_state=seed) 


    d_class_weights = None
    if class_reweighting:
     # for class reweighting
        y_integers = np.argmax(labels, axis=1)
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
        d_class_weights = dict(enumerate(class_weights))

 
    model = models_dict['m5_optuna'](X[0].shape[0], num_classes, trial)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-03)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    

    print(model.summary())

    # check = check_memory(model, False)
    # if check is False :
    #     print('model to big, returning')


    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, min_lr=5e-08, verbose=1)

    acc_callback = ConfusionMatrixCallback(x_val, y_val)

    checkpoint_filepath = OUTPUT_FOLDER + 'm5_optuna' + '_checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

    optuna_callback = optuna.integration.TFKerasPruningCallback(trial, monitor='val_accuracy')


    model.fit(x=x_tr,
              y=y_tr,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              shuffle=True,
              validation_data=(x_val, y_val),
              use_multiprocessing = True,
              callbacks = [acc_callback, reduce_lr, optuna_callback, model_checkpoint_callback]
              ,class_weight=d_class_weights)


    model.load_weights(checkpoint_filepath)
    print('=== Final accuracy ===')
    loss, acc = model.evaluate(x=x_te, y=y_te, verbose=0)
    y_pred = np.asarray(model.predict(x_te, verbose=0))
    y_pred = np.asarray(y_pred == y_pred.max(axis=1)[:, None]).astype(int)
    print(confusion_matrix(y_te.argmax(axis=1), y_pred.argmax(axis=1)))

    return acc




class PrintCallback:
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        print()
        print("Trial number: ",trial.number)
        
        if trial.state == optuna.trial.TrialState.PRUNED:      
            print('Pruned')
        else :
            print("  Value: ", trial.value)
            print("  Params: ", trial.params)

        trial_best = study.best_trial
        print("Best trial:", trial_best.number)
        print("  Value: ", trial_best.value)
        print("  Params: ", trial_best.params)
        print('---------------------------------')
        print()



if __name__ == "__main__":
    warnings.warn(
        "Recent Keras release (2.4.0) simply redirects all APIs "
        "in the standalone keras package to point to tf.keras. "
        "There is now only one Keras: tf.keras. "
        "There may be some breaking changes for some workflows by upgrading to keras 2.4.0. "
        "Test before upgrading. "
        "REF: https://github.com/keras-team/keras/releases/tag/2.4.0. "
        "There is an alternative callback function that can be used instead: "
        ":class:`~optuna.integration.TFKerasPruningCallback`",
    )
    study = optuna.create_study(direction="maximize", 
                                study_name='m5_optuna_3',
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=30))
    study.optimize(objective, n_trials=300, callbacks=[PrintCallback()])
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    joblib.dump(study, "/local/user/jrn/tinyml-challenge-2022/results/study_3.pkl")



