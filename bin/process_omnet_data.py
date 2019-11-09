#!/usr/bin/env python3
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

"""
This script processes data sets from datasets_v0 to produce the corresponding TFRecords.
"""

import os

from routenet.data_utils.omnet_tfrecord_utils_v0 import process_data

omnet_data_dir = os.getenv('OMNET_DATA_DIR')

datasets = ['nsfnetbw', 'geant2bw', 'synth50bw']

for dataset_name in datasets:
    process_data(network_data_dir=omnet_data_dir + dataset_name)
