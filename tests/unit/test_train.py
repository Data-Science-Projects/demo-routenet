# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

import glob
import os
import shutil
import unittest

import tensorflow as tf

from routenet.model.routenet_model_v0 import RouteNetModelV0
from routenet.train import train as rn_train

TEST_CODE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestTrain(unittest.TestCase):

    data_dir_root = TEST_CODE_DIR + '/../unit-resources/nsfnetbw/data/tfrecords/'

    def test_a_tfrecord_input_fn(self):
        train_files_list = glob.glob(self.data_dir_root + '/train/*.tfrecords')

        sample = rn_train.tfrecord_input_fn(filenames=train_files_list,
                                            hparams=RouteNetModelV0.default_hparams,
                                            shuffle_buf=30,
                                            target='delay')

        sample_dict_keys = sample[0].keys()

        test_dict_keys = {'traffic',
                          'sequences',
                          'link_capacity',
                          'links',
                          'paths',
                          'n_links',
                          'n_paths',
                          'n_total'}

        assert(set(sample_dict_keys) == test_dict_keys)

        assert(sample[0]['traffic'].op.type == 'ConcatV2')
        assert(sample[0]['sequences'].op.type == 'ConcatV2')
        assert(sample[0]['link_capacity'].op.type == 'ConcatV2')
        assert(sample[0]['links'].op.type == 'ConcatV2')
        assert(sample[0]['paths'].op.type == 'ConcatV2')
        assert(sample[0]['n_links'].op.type == 'AddN')
        assert(sample[0]['n_paths'].op.type == 'AddN')
        assert(sample[0]['n_total'].op.type == 'AddN')

    def test_b_train(self):

        train_files_list = glob.glob(self.data_dir_root + '/train/*.tfrecords')
        eval_files_list = glob.glob(self.data_dir_root + '/evaluate/*.tfrecords')

        model_chkpnt_dir = './model_checkpoints/'

        shutil.rmtree(model_chkpnt_dir, ignore_errors=True)

        test_checkpointing_config = tf.estimator.RunConfig(
            save_checkpoints_secs=10,  # Save checkpoints every 10 secs
            keep_checkpoint_max=20  # Retain the 20 most recent checkpoints.
        )

        rn_train.train_and_evaluate(model_dir=model_chkpnt_dir,
                                    train_files=train_files_list,
                                    shuffle_buf=3000,
                                    target='delay',
                                    train_steps=5,
                                    eval_files=eval_files_list,
                                    warm_start_from=None,
                                    checkpointing_config=test_checkpointing_config)

        assert (os.path.exists(model_chkpnt_dir + '/model.ckpt-5.data-00000-of-00001'))

        shutil.rmtree(model_chkpnt_dir, ignore_errors=True)

