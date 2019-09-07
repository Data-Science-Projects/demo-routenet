# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

import glob
import os
import unittest

from routenet.model.routenet_model import RouteNetModel
from routenet.train import train as rn_train

TEST_CODE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestTrain(unittest.TestCase):

    data_dir_root = TEST_CODE_DIR + '/../unit-resources/nsfnetbw/data/'

    def test_tfrecord_input_fn(self):
        train_files_list = glob.glob(self.data_dir_root + '/tfrecords/train/*.tfrecords')

        sample = rn_train.tfrecord_input_fn(filenames=train_files_list,
                                            hparams=RouteNetModel.default_hparams,
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
