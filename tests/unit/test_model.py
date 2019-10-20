# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

import os
import random
import unittest

from sklearn.metrics import mean_squared_error, r2_score

import utils.test_utils as test_utils

# TODO This code should be refactored with the code in the rn_notebook_utils.py

TEST_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
random.seed(13)

"""
These tests are executed against a sample of tfrecords data in the unit-resources directory.

The tfrecords ...
"""


class TestModel(unittest.TestCase):

    data_dir_root = TEST_CODE_DIR + '/../unit-resources/'
    proj_dir_root = TEST_CODE_DIR + '/../../'

    def get_sample(self, network_name):
        # Path to data sets
        train_data_path = self.data_dir_root + network_name + '/data/tfrecords/train/'
        train_data_filename = random.choice(os.listdir(train_data_path))
        sample_file = train_data_path + train_data_filename
        return sample_file

    def do_test_prediction(self, sample_file):
        graph, readout, data_set_itrtr, label = test_utils.get_model_readout(sample_file)
        median_prediction, true_val = \
            test_utils.run_predictions(graph, readout, data_set_itrtr, label, 50000,
                                       self.proj_dir_root)
        mse = mean_squared_error(median_prediction, true_val)
        r2 = r2_score(median_prediction, true_val)
        return mse, r2

    def test_a_nsfnetbw_predictions(self):
        sample_file = self.get_sample('nsfnetbw')
        mse, r2 = self.do_test_prediction(sample_file)
        assert(mse < 0.0009)
        assert(r2 > 0.98)

    def test_b_geant2bw_predictions(self):
        sample_file = self.get_sample('geant2bw')
        mse, r2 = self.do_test_prediction(sample_file)
        assert(mse < 0.0009)
        assert(r2 > 0.98)

