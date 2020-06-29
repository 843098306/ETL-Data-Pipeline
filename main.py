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

# Copy from Tensorflow Tutorial Website https://www.tensorflow.org/tfx/tutorials/transform/census
class MapAndFilterErrors(beam.PTransform):
  """Like beam.Map but filters out errors in the map_fn."""

  class _MapAndFilterErrorsDoFn(beam.DoFn):
    """Count the bad examples using a beam metric."""

    def __init__(self, fn):
      self._fn = fn
      # Create a counter to measure number of bad elements.
      self._bad_elements_counter = beam.metrics.Metrics.counter(
          'census_example', 'bad_elements')

    def process(self, element):
      try:
        yield self._fn(element)
      except Exception:  # pylint: disable=broad-except
        # Catch any exception the above call.
        self._bad_elements_counter.inc(1)

  def __init__(self, fn):
    self._fn = fn

  def expand(self, pcoll):
    return pcoll | beam.ParDo(self._MapAndFilterErrorsDoFn(self._fn))

# Copy from Tensorflow Tutorial Website https://www.tensorflow.org/tfx/tutorials/transform/census with edition adjusted for the specific use case 
def preprocessing_fn(inputs):
  """Preprocess input columns into transformed columns."""
  # Since we are modifying some features and leaving others unchanged, we
  # start by setting `outputs` to a copy of `inputs.
  outputs = inputs.copy()

  # Scale numeric columns to have range [0, 1].
  for key in NUMERIC_FEATURE_KEYS:
    outputs[key] = tft.scale_to_0_1(outputs[key])

  # for key in STRING_FEATURES_KEYS:
    # This is a SparseTensor because it is optional. Here we fill in a default
    # value when it is missing.
    # sparse = tf.sparse.SparseTensor(outputs[key].indices, outputs[key].values,
                                    # [outputs[key].dense_shape[0], 1])
    # dense = tf.sparse.to_dense(sp_input=sparse, default_value=0.)
    # Reshaping from a batch of vectors of size 1 to a batch to scalars.
    # dense = tf.squeeze(dense, axis=1)
    # outputs[key] = tft.scale_to_0_1(dense)

  # For all categorical columns except the label column, we generate a
  # vocabulary but do not modify the feature.  This vocabulary is instead
  # used in the trainer, by means of a feature column, to convert the feature
  # from a string to an integer id.
  for key in CATEGORICAL_FEATURE_KEYS:
    tft.vocabulary(inputs[key], vocab_filename=key)

  # For the label column we provide the mapping from string to index.
  # table_keys = ['>50K', '<=50K']
  # initializer = tf.lookup.KeyValueTensorInitializer(
      # keys=table_keys,
      # values=tf.cast(tf.range(len(table_keys)), tf.int64),
      # key_dtype=tf.string,
      # value_dtype=tf.int64)
  # table = tf.lookup.StaticHashTable(initializer, default_value=-1)
  # outputs[LABEL_KEY] = table.lookup(outputs[LABEL_KEY])

  return outputs
  
def transform_data(known_args, column_name_arr):
  """Transform the data and write out as a TFRecord of Example protos.

  Read in the data using the CSV reader, and transform it using a
  preprocessing pipeline that scales numeric data and converts categorical data
  from strings to int64 values indices, by creating a vocabulary for each
  category.

  Args:
    train_data_file: File containing training data
    test_data_file: File containing test data
    working_dir: Directory to write transformed data and metadata to
  """

  # The "with" block will create a pipeline, and run that pipeline at the exit
  # of the block.
  with beam.Pipeline() as pipeline:
    with tft_beam.Context(temp_dir=known_args.temp):
      # Create a coder to read the census data with the schema.  To do this we
      # need to list all columns in order since the schema doesn't specify the
      # order of columns in the csv.
      ordered_columns = column_name_arr
      converter = tft.coders.CsvCoder(ordered_columns, RAW_DATA_METADATA.schema)

      # Read in raw data and convert using CSV converter.  Note that we apply
      # some Beam transformations here, which will not be encoded in the TF
      # graph since we don't do them from within tf.Transform's methods
      # (AnalyzeDataset, TransformDataset etc.).  These transformations are just
      # to get data into a format that the CSV converter can read, in particular
      # removing spaces after commas.
      #
      # We use MapAndFilterErrors instead of Map to filter out decode errors in
      # convert.decode which should only occur for the trailing blank line.
      raw_data = (
          pipeline
          | 'ReadTrainData' >> beam.io.ReadFromText(known_args.input)
          | 'FixCommasTrainData' >> beam.Map(
              lambda line: line.replace(', ', ','))
          | 'DecodeTrainData' >> MapAndFilterErrors(converter.decode))

      # Combine data and schema into a dataset tuple.  Note that we already used
      # the schema to read the CSV data, but we also need it to interpret
      # raw_data.
      raw_dataset = (raw_data, RAW_DATA_METADATA)
      transformed_dataset, transform_fn = (
          raw_dataset | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
      transformed_data, transformed_metadata = transformed_dataset
      transformed_data_coder = tft.coders.ExampleProtoCoder(
          transformed_metadata.schema)

      _ = (
          transformed_data
          | 'EncodeTrainData' >> beam.Map(transformed_data_coder.encode)
          | 'WriteTrainData' >> beam.io.WriteToTFRecord(
              known_args.output + "_transformed"))
              


def find_feature_type_in_tuple(tuple_arr,feature_name):
    for tuple in tuple_arr:
        (feature,feature_type) = tuple
        if(feature == feature_name):
            return feature_type
    return -1
  
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
    
    
  # TRANSFORMED_TRAIN_DATA_FILEBASE = 'train_transformed'
  # TRANSFORMED_TEST_DATA_FILEBASE = 'test_transformed'
    
  transform_data(known_args,column_name_arr)
  
  filenames = [str(known_args.output + "_transformed-00000-of-00001")]
  raw_dataset = tf.data.TFRecordDataset(filenames)
  
  # for raw_record in raw_dataset.take(1):
    # example = tf.train.Example()
    # example.ParseFromString(raw_record.numpy())
    # print(example)

  def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, RAW_DATA_FEATURE_SPEC)

  parsed_dataset = raw_dataset.map(_parse_function)
  
  # Convert TFrecord Dataset into csv format
  df = pd.DataFrame(columns = column_train_feature_name)
  
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
  # model_train(df,column_train_feature_name[-1],"Titanic",2,(9,))
  model_train_hyperparam_tune(df,column_train_feature_name[-1],2,(len(column_train_feature_name) - 1,))
  model = models.load_model(str('Titanic' + ".h5"))
  for layer in model.layers:
    if len(layer.weights) > 0:
        print(layer.name, layer.weights[0].shape)

  
    
    
    