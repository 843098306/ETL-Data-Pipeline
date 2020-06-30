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

from train_and_hyperparam_tune import *
from datatype_reference import *        

# Some util functions
def find_feature_type_in_tuple(tuple_arr,feature_name):
    for tuple in tuple_arr:
        (feature,feature_type) = tuple
        if(feature == feature_name):
            return feature_type
    return -1
    
def extract_column_names(datatype_array,label_name):
    category = []
    numeric = []
    string_arr = []
    
    for i in datatype_array:
        (column_name,columntype) = i
        if(column_name != label_name):
            if(columntype == 'Boolean' or columntype == 'Categorical'):
                category.append(column_name)
            elif(columntype == 'String'):
                string_arr.append(column_name)
            else:
                numeric.append(column_name)
        else:
            label = (column_name,columntype)
            
    return category,numeric,string_arr,label

def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, RAW_DATA_FEATURE_SPEC)
  
if __name__ == '__main__':
  # logging.getLogger().setLevel(logging.INFO)
  known_args = data_type_reference()
  datatype_filepath = str(known_args.output + "-00000-of-00001")
  
  datatype_arr_temp = []
  char_list = ["'","(",")"]
  
  with open(datatype_filepath, "rb") as fp:
    for i in fp.readlines():
        tmp = i.decode().strip().split(",")
        tmp[0] = str(tmp[0][2:-1])
        tmp[1] = str(tmp[1][2:-2])
        datatype_arr_temp.append((tmp[0], tmp[1]))
  
  # Get label column name and column name array
  raw_header = pd.read_csv(known_args.input,header=None,nrows=1)
  column_name_arr = []
  for i in range(raw_header.shape[1]):
    column_name_arr.append(raw_header.iloc[0,i])
    
  label_name = raw_header.iloc[0,-1]
  
  # Make sure that features in datatype_arr array and column_name_arr array are in the same order
  datatype_arr = []
  for i in range(len(column_name_arr)):
    feature = column_name_arr[i]
    datatype_arr.append((feature,find_feature_type_in_tuple(datatype_arr_temp,feature)))
    
  # Make the column arrays that we use for input, typically that contain only categorical or numerica data  
  column_train_feature_name = []
  for i in range(len(datatype_arr)):
    (feature_name,feature_type) = datatype_arr[i]
    if(feature_type == 'Boolean' or feature_type == 'Categorical' or feature_type == "Numeric"):
        column_train_feature_name.append(feature_name)
    
  print(column_name_arr)
  print(datatype_arr)
  print(column_train_feature_name)
  
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
  
  # Convert TFrecord Dataset into csv format
  df = pd.DataFrame(columns = column_train_feature_name)
  parsed_dataset = raw_dataset.map(_parse_function)
  
  for feature_row in parsed_dataset:
    temp = {}
    i = 0
    for feature in column_name_arr:
        # Only Categorical and numerica data works here. Filter out String Data
        try:
            feature_values = feature_row[feature].numpy()
            temp[column_train_feature_name[i]] = feature_values
            i += 1
            if(i == len(column_train_feature_name)):
                i = 0
        except:
            pass
    df = df.append(temp, ignore_index=True)
  
  print(df.head(1))
  # model_train_hyperparam_tune(df,column_train_feature_name[-1],2,(len(column_train_feature_name) - 1,))
  model = models.load_model(str('Titanic' + ".h5"))
  for layer in model.layers:
    if len(layer.weights) > 0:
        print(layer.name, layer.weights[0].shape)

  
    
    
    