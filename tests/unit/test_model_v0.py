# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

import os
import random
import unittest

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
        # TODO change to evaluate, rather than train?
        train_data_path = self.data_dir_root + network_name + '/data/tfrecords/train/'
        train_data_filename = random.choice(os.listdir(train_data_path))
        sample_file = train_data_path + train_data_filename
        return sample_file

    def test_a_nsfnetbw_predictions(self):
        sample_file = self.get_sample('nsfnetbw')
        mse, r2 = test_utils.do_test_prediction(sample_file, self.proj_dir_root +
                                                'model_checkpoints_nsfnetbw_synth50bw_50000_v0',
                                                normalise_pred=True)
        assert(mse < 0.0009)
        assert(r2 > 0.98)

    def test_b_geant2bw_predictions(self):
        sample_file = self.get_sample('geant2bw')
        mse, r2 = test_utils.do_test_prediction(sample_file,
                                                self.proj_dir_root +
                                                'model_checkpoints_nsfnetbw_synth50bw_50000_v0',
                                                normalise_pred=True)
        assert(mse < 0.0009)
        assert(r2 > 0.98)
