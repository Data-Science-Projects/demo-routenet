# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

import glob
import os
import random
import shutil
import sys
import unittest

import tensorflow as tf

import routenet.data_utils.omnet_tfrecord_utils_v0
import utils.test_utils as test_utils
from routenet.train import train_v0 as rn_train
from routenet.train.normvals_v0 import NormVals

TEST_CODE_DIR = os.path.dirname(os.path.abspath(__file__))


class SmokeTest(unittest.TestCase):
    data_dir_root = TEST_CODE_DIR + '/../smoke-resources/data/'
    checkpoint_dir = TEST_CODE_DIR + '/../smoke-resources/CheckPoints/'

    random.seed(13)

    def do_test_tfrecords(self, data_set_name):
        routenet.data_utils.omnet_tfrecord_utils_v0.process_data(network_data_dir=self.data_dir_root +
                                                                                  data_set_name)
        test_dirs = glob.glob(self.data_dir_root + data_set_name + '/tfrecords/*')
        assert self.data_dir_root + data_set_name + '/tfrecords/evaluate' in test_dirs
        assert self.data_dir_root + data_set_name + '/tfrecords/train' in test_dirs

    def test_1_nsfnetbw_tfrecords(self):
        data_set_name = 'nsfnetbw'
        self.do_test_tfrecords(data_set_name)

    """
    def test_1_geant2bw_tfrecords(self):
        data_set_name = 'geant2bw'
        self.do_test_tfrecords(data_set_name)

    def test_1_synth50bw_tfrecords(self):
        data_set_name = 'synth50bw'
        self.do_test_tfrecords(data_set_name)
    """

    def do_train(self, network_name, train_steps=100):
        train_files_list = glob.glob(self.data_dir_root + network_name +
                                     '/tfrecords/train/*.tfrecords')
        eval_files_list = glob.glob(self.data_dir_root + network_name +
                                    '/tfrecords/evaluate/*.tfrecords')

        test_checkpointing_config = tf.estimator.RunConfig(
            save_checkpoints_secs=10,  # Save checkpoints every 10 secs
            keep_checkpoint_max=20  # Retain the 20 most recent checkpoints.
        )

        norm_vals = NormVals()
        #norm_vals.calculate_norm_vals([network_name])
        norm_vals.read_norm_params(self.checkpoint_dir + '/../')

        rn_train.train_and_evaluate(model_dir=self.checkpoint_dir + network_name,
                                    train_files=train_files_list,
                                    shuffle_buf=30000,
                                    target='delay',
                                    train_steps=train_steps,
                                    eval_files=eval_files_list,
                                    warm_start_from=None,
                                    checkpointing_config=test_checkpointing_config,
                                    norm_vals=norm_vals)
        norm_vals.save_norm_params(self.checkpoint_dir + network_name)

    def test_2_nsfnetbw_train(self):
        self.do_train('nsfnetbw', train_steps=200)

    """
    def test_2_geant2bw_train(self):
        self.do_train('geant2bw', train_steps=200)

    def test_2_synth50bw_train(self):
        self.do_train('synth50bw', train_steps=500)
    """

    def get_sample(self, network_name):
        # Path to data sets
        train_data_path = self.data_dir_root + network_name + '/tfrecords/train/'
        train_data_filename = random.choice(os.listdir(train_data_path))
        sample_file = train_data_path + train_data_filename
        return sample_file

    def do_pred_test(self, network_name, checkpoint_id=100):
        sample_file = self.get_sample(network_name)
        norm_vals = NormVals()
        norm_vals.read_norm_params(self.checkpoint_dir + '/../')
        #norm_vals.read_norm_params(path=self.checkpoint_dir + network_name)
        mse, r2 = test_utils.do_test_prediction(sample_file,
                                                checkpoint_dir=self.checkpoint_dir + network_name,
                                                checkpoint_id=checkpoint_id,
                                                norm_vals=norm_vals)

        print('For {0}, we have mse {1} and r2 {2}'.format(network_name, mse, r2), file=sys.stderr)
        assert(mse < 0.05)
        assert(r2 > 0.5)

    def test_3_nsfnetbw_predictions(self):
        self.do_pred_test('nsfnetbw', checkpoint_id=200)

    """
    def test_3_geant2bw_predictions(self):
        self.do_pred_test('geant2bw', checkpoint_id=200)

    def test_3_synth50bw_predictions(self):
        self.do_pred_test('synth50bw', checkpoint_id=500)
    """

    @classmethod
    def do_teardown(cls, data_set_name):
        shutil.rmtree(cls.data_dir_root + '/../CheckPoints/' + data_set_name,
                      ignore_errors=True)
        shutil.rmtree(cls.data_dir_root + data_set_name + '/tfrecords', ignore_errors=True)

    @classmethod
    def tearDownClass(cls):
        cls.do_teardown('nsfnetbw')
        cls.do_teardown('geant2bw')
        cls.do_teardown('synth50bw')
