# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

import glob
import os
import random
import shutil
import unittest

import numpy as np
import tensorflow as tf

import routenet.data_utils.omnet_tfrecord_utils
from routenet.model.routenet_model import RouteNetModel
from routenet.train import train as rn_train
from routenet.train.train import read_dataset

TEST_CODE_DIR = os.path.dirname(os.path.abspath(__file__))


class SmokeTest(unittest.TestCase):
    # TODO use environment variable for data dir

    data_dir_root = TEST_CODE_DIR + '/../smoke-resources/data/'

    random.seed(13)

    def do_test_tfrecords(self, data_set_name):
        routenet.data_utils.omnet_tfrecord_utils.process_data(network_data_dir=self.data_dir_root +
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

    def do_train(self, data_set_name):
        train_files_list = glob.glob(self.data_dir_root + data_set_name +
                                     '/tfrecords/train/*.tfrecords')
        eval_files_list = glob.glob(self.data_dir_root + data_set_name +
                                    '/tfrecords/evaluate/*.tfrecords')

        rn_train.train_and_evaluate(model_dir=TEST_CODE_DIR +
                                    ('/../smoke-resources/CheckPoints/' + data_set_name),
                                    train_files=train_files_list,
                                    shuffle_buf=30,
                                    target='delay',
                                    train_steps=100,
                                    eval_files=eval_files_list,
                                    warm_start_from=None)

    def test_2_nsfnetbw_train(self):
        self.do_train('nsfnetbw')

    """
    def test_2_geant2bw_train(self):
        self.do_train('geant2bw')

    def test_2_synth50bw_train(self):
        self.do_train('synth50bw')
    """

    def do_predict(self, data_set_name):
        omnet_data_dir = os.getenv('OMNET_DATA_DIR')
        train_data_path = omnet_data_dir + '/' + data_set_name + '/tfrecords/train/'
        train_data_filename = random.choice(os.listdir(train_data_path))
        sample_file = train_data_path + train_data_filename

        graph_predict = tf.Graph()
        with graph_predict.as_default():
            model = RouteNetModel()
            model.build()

            data_set = read_dataset(sample_file, target='delay')
            itrtr = data_set.make_initializable_iterator()
            features, label = itrtr.get_next()

            with tf.name_scope('predict'):
                predictions = tf.map_fn(lambda x: model(x, training=True),
                                        features, dtype=tf.float32)

            preds = tf.squeeze(predictions)
            # This is in the inverse of the transform applied in tfrecords_utils.parse()
            # Why the transform is applied at all is unclear though. TODO
            predictions = 0.54 * preds + 0.37

        with tf.compat.v1.Session(graph=graph_predict) as sess:
            sess.run(tf.compat.v1.local_variables_initializer())
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            # path to the checkpoint we want to restore
            # TODO replace with newly trained model checkpoint from test?
            saver.restore(sess, self.data_dir_root + '/../CheckPoints/'
                          + data_set_name + '/model.ckpt-100')

            hats = []
            labels = []
            for _ in range(50):
                sess.run(itrtr.initializer)
                pred_delay, label_delay = sess.run([predictions, label])
                hats.append(pred_delay)
                labels.append(label_delay)

            final_prediction = np.median(hats, axis=0)
            # test_final_prediction = np.genfromtxt('../smoke-resources/test_final_prediction.csv',
            #                                      delimiter=',')
            # print(final_prediction)
            # print(test_final_prediction)
            # assert (final_prediction == test_final_prediction)

    def test_3_nsfnetbw_predict(self):
        self.do_predict('nsfnetbw')

    """
    def test_3_geant2bw_predict(self):
        self.do_predict('geant2bw')

    def test_3_synth50bw_predict(self):
        self.do_predict('synth50bw')
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
