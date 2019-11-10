#!/usr/bin/env python3
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

"""
This script calculates how many results records there are for datasets from datasets_v0.
"""

import os

from routenet.data_utils.omnet_tfrecord_utils_v0 import count_data

omnet_data_dir = os.getenv('OMNET_DATA_DIR')

datasets = ['nsfnetbw', 'geant2bw', 'synth50bw']

for dataset in datasets:
    count = count_data(omnet_data_dir + '/datasets_v0/' + dataset)
    print(dataset + ' - ' + str(count) + '\n')
