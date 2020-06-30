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
    
def column_operation(datatype_filepath, known_args):
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
    
    return column_name_arr,datatype_arr,column_train_feature_name,label_name
    
def Convert_TFrecord_to_PandasDF(parsed_dataset,column_train_feature_name,column_name_arr):
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
        
    return df