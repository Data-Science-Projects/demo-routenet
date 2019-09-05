# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

import os
import unittest

TEST_CODE_DIR = os.path.dirname(os.path.abspath(__file__))


class TrainTest(unittest.TestCase):

    def test_tfrecord_input_fn(self):
        pass
        """
        rn_train.tfrecord_input_fn(filenames=train_files,
                          hparams=model_hparams,
                          shuffle_buf=shuffle_buf,
                          target=target)
        """