#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

from utils import *
from train_and_hyperparam_tune import *
from datatype_reference import * 

def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, RAW_DATA_FEATURE_SPEC)
  
if __name__ == '__main__':
  # logging.getLogger().setLevel(logging.INFO)
  known_args = data_type_reference()
  datatype_filepath = str(known_args.output + "-00000-of-00001")
  
  column_name_arr,datatype_arr,column_train_feature_name,label_name = column_operation(datatype_filepath = datatype_filepath,known_args = known_args)
  
  # Extract column names and put them in different lists based on their feature types
  CATEGORICAL_FEATURE_KEYS,NUMERIC_FEATURE_KEYS,STRING_FEATURES_KEYS,label = extract_column_names(datatype_arr,label_name)
  (LABEL_KEY,LABEL_TYPE) = label
  print(CATEGORICAL_FEATURE_KEYS)
  print(NUMERIC_FEATURE_KEYS)
  print(STRING_FEATURES_KEYS)
  print(LABEL_KEY)
  
  # Define data schema
  RAW_DATA_FEATURE_SPEC = dict(
    [(name, tf.io.FixedLenFeature([], tf.string))
     for name in CATEGORICAL_FEATURE_KEYS] +
    [(name, tf.io.FixedLenFeature([], tf.float32))
     for name in NUMERIC_FEATURE_KEYS] +
    [(name, tf.io.VarLenFeature(tf.string))
     for name in STRING_FEATURES_KEYS] +
    [(LABEL_KEY, tf.io.FixedLenFeature([], tf.float32))])

  RAW_DATA_METADATA = tft.tf_metadata.dataset_metadata.DatasetMetadata(
    tft.tf_metadata.dataset_schema.schema_utils.schema_from_feature_spec(RAW_DATA_FEATURE_SPEC))
    
  # RAW_DATA_METADATA and other variables can be passed into the data_transform script
  from data_transform import *
    
  transform_data(known_args,column_name_arr)
  
  filenames = [str(known_args.output + "_transformed-00000-of-00001")]
  raw_dataset = tf.data.TFRecordDataset(filenames)
  
  # Read example row of record in the dataset 
  # for raw_record in raw_dataset.take(1):
    # example = tf.train.Example()
    # example.ParseFromString(raw_record.numpy())
    # print(example)
  
  parsed_dataset = raw_dataset.map(_parse_function)
  
  # Convert TFrecord Dataset into csv format
  df = Convert_TFrecord_to_PandasDF(parsed_dataset,column_train_feature_name,column_name_arr)
  
  # Train the model using Keras and tune the hyperparameters of the model using Keras tuner
  model_train_hyperparam_tune(df,column_train_feature_name[-1],2,(len(column_train_feature_name) - 1,))
  
  model = models.load_model(str('Titanic' + ".h5"))
  for layer in model.layers:
    if len(layer.weights) > 0:
        print(layer.name, layer.weights[0].shape)

  
    
    
    