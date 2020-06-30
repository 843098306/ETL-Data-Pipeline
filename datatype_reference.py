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

def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

class Split(beam.DoFn):
    def process(self, element, header_array):
        Feature_list_raw = element.split(",")
        Feature_list = []
        i = 0
        while(i < len(Feature_list_raw)):
            if(i == 2):
                Feature_list.append(str(Feature_list_raw[i] + Feature_list_raw[i + 1]))
                i += 2
            else:
                Feature_list.append(Feature_list_raw[i])
                i += 1
            
        Output = {}
        for j in range(len(header_array)):
            if(Feature_list[j] == "" or Feature_list[j] == " "):
                Output[header_array[j]] = "?"
            else:
                Output[header_array[j]] = Feature_list[j]
        return [Output]
        
        
class Collect(beam.DoFn):
    def process(self, element):
        # Returns a list of tuples containing feature, feature values and feature type
        result = []
        for feature in element:
            if(isfloat(element[feature])):
                result.append((feature,element[feature],'Numeric'))
            elif(element[feature] != "?"):
                result.append((feature,element[feature],'String'))
                
        return result

def data_type_reference(argv=None, save_main_session=True):
  """Main entry point; defines and runs the wordcount pipeline."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input',
      dest='input',
      required=True,
      help='Input file to process.')
  parser.add_argument(
      '--output',
      dest='output',
      required=True,
      help='Output file to write results to.')
  parser.add_argument(
      '--temp',
      dest='temp',
      required=True,
      help='Temp file')
  known_args, pipeline_args = parser.parse_known_args(argv)
  print(known_args.input,known_args.output)
  
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
  p = beam.Pipeline(options=pipeline_options)
  
  raw_header = pd.read_csv(known_args.input,header=None,nrows=1)
  print(raw_header)
  header_array = []
  for i in range(raw_header.shape[1]):
    header_array.append(raw_header.iloc[0,i])
  
  csv_lines = (
    p | beam.io.ReadFromText(known_args.input,skip_header_lines=1) |
    beam.ParDo(Split(),header_array)
  )
  
  def distinct_count_extract(l):
    (tuple_list, number) = l
    return (tuple_list[0], (1,tuple_list[2]))
    
  def count_values(value_ones):
    (feature,arr) = value_ones
    ones = [i[0] for i in arr]
    type = [i[1] for i in arr]
    return (feature, sum(ones),max(type, key = type.count))
    
  def feature_reference(value_counts):
    (index,counts,type) = value_counts
    print((index,counts,type))
    if(counts == 2):
        return (index,"Boolean")
    elif(counts > 2 and counts <= 15):
        return (index,"Categorical")
    elif(type == "Numeric"):
        return (index,"Numeric")
    return (index,"String")
    
  type_reference = (
  csv_lines | beam.ParDo(Collect()) |
  "PerElement Count" >> beam.combiners.Count.PerElement() |
  "Distinct Count Preprocess" >> beam.Map(distinct_count_extract) |
  "Distinct Count Group" >> beam.GroupByKey() |
  "Distinct Count" >> beam.Map(count_values) |
  "Map Result" >> beam.Map(feature_reference)
  )

  output = (
    type_reference | beam.io.WriteToText(known_args.output)
  )
  
  result = p.run()
  result.wait_until_finish()
  
  return known_args


