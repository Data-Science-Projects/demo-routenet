# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

"""
Derived from normalise.py
Copyright (c) 2019, Paul Almasan, Sergi Carol.
Universitat Politècnica de Catalunya, Computer Architecture department, Barcelona, Spain
"""

import os
import random

import numpy as np
import tensorflow as tf


class NormVals:
    mean_delay = 0
    std_delay = 1
    max_delay = 0
    min_delay = 0

    mean_jitter = 0
    std_jitter = 1
    max_jitter = 0
    min_jitter = 0

    mean_traffic = 0
    std_traffic = 1
    max_traffic = 0
    min_traffic = 0

    mean_link_capacity = 0
    std_link_capacity = 1
    max_link_capacity = 0
    min_link_capacity = 0

    def save_norm_params(self, path='./', filename='norm.params'):

        with open(path + '/' + filename, 'w') as params_file:
            params_file.write('mean_delay=' + str(self.mean_delay) + '\n' +
                              'std_delay=' + str(self.std_delay) + '\n' +
                              'max_delay=' + str(self.max_delay) + '\n' +
                              'min_delay=' + str(self.min_delay) + '\n' +
                              'mean_jitter=' + str(self.mean_jitter) + '\n' +
                              'std_jitter=' + str(self.std_jitter) + '\n' +
                              'max_jitter=' + str(self.max_jitter) + '\n' +
                              'min_jitter=' + str(self.min_jitter) + '\n' +
                              'mean_traffic=' + str(self.mean_traffic) + '\n' +
                              'std_traffic=' + str(self.std_traffic) + '\n' +
                              'max_traffic=' + str(self.max_traffic) + '\n' +
                              'min_traffic=' + str(self.min_traffic) + '\n' +
                              'mean_link_capacity=' + str(self.mean_link_capacity) + '\n' +
                              'std_link_capacity=' + str(self.std_link_capacity) + '\n' +
                              'max_link_capacity=' + str(self.max_link_capacity) + '\n' +
                              'min_link_capacity=' + str(self.min_link_capacity))

    def read_norm_params(self, path='./', filename='norm.params'):

        with open(path + '/' + filename) as params_file:
            for line in params_file:
                if 'mean_delay' in line:
                    self.mean_delay = float(line.split('=')[1])
                if 'std_delay' in line:
                    self.std_delay = float(line.split('=')[1])
                if 'max_delay' in line:
                    self.max_delay = float(line.split('=')[1])
                if 'min_delay' in line:
                    self.min_delay = float(line.split('=')[1])
                if 'mean_jitter' in line:
                    self.mean_jitter = float(line.split('=')[1])
                if 'std_jitter' in line:
                    self.std_jitter = float(line.split('=')[1])
                if 'max_jitter' in line:
                    self.max_jitter = float(line.split('=')[1])
                if 'min_jitter' in line:
                    self.min_jitter = float(line.split('=')[1])
                if 'mean_traffic' in line:
                    self.mean_traffic = float(line.split('=')[1])
                if 'std_traffic' in line:
                    self.std_traffic = float(line.split('=')[1])
                if 'max_traffic' in line:
                    self.max_traffic = float(line.split('=')[1])
                if 'min_traffic' in line:
                    self.min_traffic = float(line.split('=')[1])
                if 'mean_link_capacity' in line:
                    self.mean_link_capacity = float(line.split('=')[1])
                if 'std_link_capacity' in line:
                    self.std_link_capacity = float(line.split('=')[1])
                if 'max_link_capacity' in line:
                    self.max_link_capacity = float(line.split('=')[1])
                if 'min_link_capacity' in line:
                    self.min_link_capacity = float(line.split('=')[1])

    def calculate_norm_vals(self, dataset_names, sample_size=20):
        delays = []
        traffic = []
        link_capacity = []
        jitter = []

        omnet_data_dir = os.getenv('OMNET_DATA_DIR')

        for dataset in dataset_names:
            for tfrecords in ['train', 'evaluate']:
                tfrecords_dir = omnet_data_dir + '/datasets_v0/' + dataset + '/tfrecords/' + tfrecords
                files = os.listdir(tfrecords_dir)
                sample = random.sample(files, sample_size)
                for file in sample:
                    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_dir + '/' + file)
                    for string_record in record_iterator:
                        example = tf.train.Example()
                        example.ParseFromString(string_record)
                        delays += example.features.feature['delay'].float_list.value
                        traffic += example.features.feature['traffic'].float_list.value
                        link_capacity += example.features.feature['link_capacity'].float_list.value
                        jitter += example.features.feature['jitter'].float_list.value

        self.mean_delay = round(np.mean(delays), 2)
        self.std_delay = round(np.std(delays), 2)
        self.max_delay = round(np.max(delays), 3)
        self.min_delay = round(np.min(delays), 3)

        self.mean_jitter = round(np.mean(jitter), 2)
        self.std_jitter = round(np.std(jitter), 2)
        self.max_jitter = round(np.max(jitter), 3)
        self.min_jitter = round(np.min(jitter), 3)

        self.mean_traffic = round(np.mean(traffic), 2)
        self.std_traffic = round(np.std(traffic), 2)
        self.max_traffic = round(np.max(traffic), 3)
        self.min_traffic = round(np.min(traffic), 3)

        self.mean_link_capacity = round(np.mean(link_capacity), 2)
        self.std_link_capacity = round(np.std(link_capacity), 2)
        self.max_link_capacity = np.max(link_capacity)
        self.min_link_capacity = np.min(link_capacity)