# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

import os
import random
import unittest

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

from routenet.model.routenet_model import RouteNetModel
from routenet.train.train import read_dataset

# TODO This code should be refactored with the code in the rn_notebook_utils.py

TEST_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
random.seed(13)


class TestModel(unittest.TestCase):

    data_dir_root = TEST_CODE_DIR + '/../unit-resources/'
    proj_dir_root = TEST_CODE_DIR + '/../../'

    def get_sample(self, network_name):
        # Path to data sets
        train_data_path = self.data_dir_root + network_name + '/data/tfrecords/train/'
        train_data_filename = random.choice(os.listdir(train_data_path))
        sample_file = train_data_path + train_data_filename
        # print(sample_file.split('/')[-1])
        return sample_file

    def get_model_readout(self, test_sample_file, normalise_readout=True):
        graph = tf.Graph()
        with graph.as_default():
            model = RouteNetModel()
            model.build()

            data_set = read_dataset(test_sample_file)
            data_set_itrtr = data_set.make_initializable_iterator()
            # The `label` here is the delay value associated with the features. The features are
            # selected in the transformation_func(...) from the train module.
            features, label = data_set_itrtr.get_next()

            with tf.name_scope('predict'):
                # The lamba construct below invokes RouteNetModel.call(features, training=True).
                # The return value from the call(...) function is the readout Sequential() model,
                # with training set to `True`.
                readout = tf.map_fn(lambda x: model(x, training=True), features, dtype=tf.float32)

            # Having called this on one set of features, we have an initialised readout.
            # We squeeze the tensor to ... TODO
            readout = tf.squeeze(readout)
            # This is the reverse of the normalisation applied in the parse function in
            # omet_tfrecord_utils.
            # TODO the rationale for the normalisation has to be explained.
            if normalise_readout:
                readout = 0.54 * readout + 0.37

            return graph, readout, data_set_itrtr, label

    def run_predictions(self, graph, readout, data_itrtr, true_value, checkpoint_id,
                        normalised_delay=True):
        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(tf.compat.v1.local_variables_initializer())
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            # Load the weights from the checkpoint
            if normalised_delay:
                saver.restore(sess, self.proj_dir_root +
                              '/model_checkpoints-with_delay_norm/model.ckpt-' +
                              str(checkpoint_id))
            else:
                saver.restore(sess, self.proj_dir_root +
                              '/model_checkpoints-no_delay_norm/model.ckpt-' +
                              str(checkpoint_id))

            # We are going to take a median of a number of predictions
            predictions = []
            # We run the model 50 times to predict delays based for the network represented by
            # the sample data set.
            for _ in range(50):
                sess.run(data_itrtr.initializer)
                # The `true_delay` value here is the original delay value from the sample data set,
                # against which we compare the median value of the predicted delay below.
                # TODO check why true_value has to be passed in
                # Note that we need to pass back the median of the `pred_delay` and the true_delay
                # just so that we have two tensors of the same shape for graphing purposes.
                predicted, true_val = sess.run([readout, true_value])
                predictions.append(predicted)

            median_prediction = np.median(predictions, axis=0)

            return median_prediction, true_val[0]

    def test_a_nsfnetbw_predictions(self):
        sample_file = self.get_sample('nsfnetbw')
        graph, readout, data_set_itrtr, label = self.get_model_readout(sample_file)
        median_prediction, true_val = \
            self.run_predictions(graph, readout, data_set_itrtr, label, 54809)
        mse = mean_squared_error(median_prediction, true_val)
        assert(mse < 0.00035)
        r2 = r2_score(median_prediction, true_val)
        assert(r2 > 0.98)
