# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

import glob
import random
import shutil
import unittest

import numpy as np
import tensorflow as tf

import routenet.data_utils.tfrecord_utils
from routenet.data_utils.tfrecord_utils import read_dataset
from routenet.model.comnet_model import ComnetModel
from routenet.train import train


class SmokeTest(unittest.TestCase):

    # TODO use environment variable for data dir
    
    data_dir_root = '../smoke-resources/data/'
    
    default_hparams = tf.contrib.training.HParams(link_state_dim=32,
                                                  path_state_dim=32,
                                                  batch_size=32,
                                                  T=8,
                                                  readout_units=256,
                                                  learning_rate=0.001,
                                                  dropout_rate=0.5,
                                                  l2=0.1,
                                                  l2_2=0.01,
                                                  learn_embedding=True)

    random.seed(13)

    def do_test_tfrecords(self, data_set_name):
        routenet.data_utils.tfrecord_utils.process_data(network_data_dir=self.data_dir_root +
                                                                         data_set_name)
        test_dirs = glob.glob(self.data_dir_root+data_set_name+'/tfrecords/*')
        assert self.data_dir_root+data_set_name+'/tfrecords/evaluate' in test_dirs
        assert self.data_dir_root+data_set_name+'/tfrecords/train' in test_dirs

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

        train.process_train(model_hparams=self.default_hparams,
                            model_dir='../smoke-resources/CheckPoints/'+data_set_name,
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

        # Path to downloaded datasets

        test_file = (self.data_dir_root + data_set_name +
                     '/tfrecords/train/results_' + data_set_name + '_9_Routing_SP_k_1.tfrecords')

        graph_predict = tf.Graph()
        with graph_predict.as_default():
            model = ComnetModel(self.default_hparams)
            model.build()

            it = read_dataset(test_file, target='delay')
            features, label = it.get_next()

            with tf.name_scope('predict'):
                predictions = tf.map_fn(lambda x: model(x, training=True),
                                        features, dtype=tf.float32)

            preds = tf.squeeze(predictions)
            # TODO Why?
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
                sess.run(it.initializer)
                pred_delay, label_delay = sess.run([predictions, label])
                hats.append(pred_delay)
                labels.append(label_delay)

            final_prediction = np.median(hats, axis=0)
            #test_final_prediction = np.genfromtxt('../smoke-resources/test_final_prediction.csv',
            #                                      delimiter=',')
            #print(final_prediction)
            #print(test_final_prediction)
            #assert (final_prediction == test_final_prediction)

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
