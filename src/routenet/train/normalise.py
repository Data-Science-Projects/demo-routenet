# -*- coding: utf-8 -*-
# Copyright (c) 2019, Paul Almasan [^1], Sergi Carol [^1], Nathan Sowatskey, Ana Matute.
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: almasan@ac.upc.edu

import os
import random

import numpy as np
import tensorflow as tf

from routenet.train.train import NormVals


def get_norm_vals(dataset_names):
    limit = 20  # Number of files to sample
    delays = []
    traffic = []
    link_capacity = []

    omnet_data_dir = os.getenv('OMNET_DATA_DIR')

    for dataset in dataset_names:
        for tfrecords in ['train', 'evaluate']:
            tfrecords_dir = omnet_data_dir + '/datasets_v0/' + dataset + '/tfrecords/' + tfrecords
            files = os.listdir(tfrecords_dir)
            sample = random.sample(files, limit)
            for file in sample:
                record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_dir + '/' + file)
                for string_record in record_iterator:
                    example = tf.train.Example()
                    example.ParseFromString(string_record)
                    delays += example.features.feature['delay'].float_list.value
                    traffic += example.features.feature['traffic'].float_list.value
                    link_capacity += example.features.feature['link_capacity'].float_list.value

    norm_vals = NormVals()

    norm_vals.mean_delay = round(np.mean(delays), 2)
    norm_vals.std_delay = round(np.std(delays), 2)
    norm_vals.mean_traffic = round(np.mean(traffic), 2)
    norm_vals.std_traffic = round(np.std(traffic), 2)
    norm_vals.mean_link_capacity = round(np.mean(link_capacity), 2)
    norm_vals.std_link_capacity = round(np.std(link_capacity), 2)

    return norm_vals

