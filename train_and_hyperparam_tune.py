from __future__ import absolute_import
from __future__ import print_function

import h5py
import argparse
import logging
import re
import numpy as np
import pandas as pd
import os


import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.metrics import Metrics
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
import keras
from keras import models, layers
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband
from sklearn import preprocessing

# Ordial Encode of Categorical Value for input
def prepare_inputs(X):
	oe = preprocessing.OrdinalEncoder()
	oe.fit(X)
	Output = oe.transform(X)
	return Output

class MyHyperModel(HyperModel):

    def __init__(self, input_shape, num_classes):
        self.num_classes = num_classes
        self.input_shape = input_shape
        if(num_classes == 2):
            self.val_acc = "val_binary_accuracy"
        else:
            self.val_acc = "val_sparse_categorical_accuracy"
        
    def build(self, hp):
        # model build with hyperparameter tuning on all layers. Can be customized
        model = keras.Sequential()
        
        model.add(layers.Dense(units=hp.Int('units',
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                                            input_shape=self.input_shape))
        model.add(layers.normalization.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(
            layers.Dropout(
                rate=hp.Float(
                    "dropout_1", min_value=0.0, max_value=0.5, default=0.25, step=0.01,
                )
            )
        )
        
        
        model.add(layers.Dense(units=hp.Int('units',
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                                            input_shape=self.input_shape))
        model.add(layers.normalization.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(
            layers.Dropout(
                rate=hp.Float(
                    "dropout_1", min_value=0.0, max_value=0.5, default=0.25, step=0.01,
                )
            )
        )
        
        model.add(layers.Dense(units=hp.Int('units',
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                                            input_shape=self.input_shape))
        model.add(layers.normalization.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(
            layers.Dropout(
                rate=hp.Float(
                    "dropout_1", min_value=0.0, max_value=0.5, default=0.25, step=0.01,
                )
            )
        )
        
        model.add(layers.Dense(units=hp.Int('units',
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                                            input_shape=self.input_shape))
        model.add(layers.normalization.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(
            layers.Dropout(
                rate=hp.Float(
                    "dropout_1", min_value=0.0, max_value=0.5, default=0.25, step=0.01,
                )
            )
        )

        # binary classification or multiclass classificatio based on number of classes
        if(self.num_classes == 2):
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4,1e-5])),
                    loss='binary_crossentropy',
                    metrics=[keras.metrics.binary_accuracy])
            return model
            
        else:
            model.add(layers.Dense(1, activation='softmax'))
            model.compile(optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4,1e-5])),
                    loss='sparse_categorical_crossentropy',
                    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])
            return model
        


def model_train_hyperparam_tune(df,column_train_feature_name,num_classes,input_shape,
    seed = 1, exe_per_trial = 2, max_epochs = 40, n_epoch_search = 100, output_dir = 'hyperparameter',project_name = 'Titanic'):
    x = df.drop([column_train_feature_name], axis=1)
    x = prepare_inputs(x)
    y = df[column_train_feature_name]
    # reshape(-1,1) makes it a valid input for the model.
    y = y.values.reshape(-1,1)
    hypermodel = MyHyperModel(num_classes = num_classes,input_shape = input_shape)
  
    tuner = Hyperband(
        hypermodel,
        objective = hypermodel.val_acc,
        seed = seed,
        executions_per_trial = exe_per_trial,
        directory = output_dir,
        project_name = project_name,
        max_epochs = max_epochs
    )
    
    tuner.search_space_summary()

    tuner.search(x, y, epochs = n_epoch_search, validation_split=0.2)
  
    # Show a summary of the search
    tuner.results_summary()

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(str(project_name + ".h5"))
    