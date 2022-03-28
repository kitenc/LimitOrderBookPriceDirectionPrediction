import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os 
from  deeplob_model import create_deeplob

from utils import train_val_split, generate_x_y

from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.model_selection import train_test_split


import tensorflow as tf
import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, LSTM, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam 
from keras.layers.advanced_activations import LeakyReLU

#warnings.filterwarnings("ignore")

full_data = pd.read_parquet('full_data_for_cnn_lstm.parq', engine='pyarrow')

# get arguments

def train(tk, k, T, full_data, feature_final, save_path):
    X, y = generate_x_y(full_data, tk, k, T, feature_final)
    X_train, X_val = train_val_split(X)
    y_train, y_val = train_val_split(y)


    deeplob = create_deeplob(X_train.shape[1], X_train.shape[2], n_hiddens)
    deeplob.summary()

    checkpoint_filepath = './cnnlstm_model_weights'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=True)

    deeplob.fit(X_train, y_train, validation_data=(X_val, y_val), 
            epochs=200, batch_size=128, verbose=2, callbacks=[model_checkpoint_callback])

    deeplob.save(save_path)







