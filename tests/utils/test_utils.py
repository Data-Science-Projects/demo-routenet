# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

from routenet.model.routenet_model import RouteNetModel
from routenet.train.train import read_dataset


# TODO This code should be refactored with the code in the rn_notebook_utils.py

def get_model_readout(test_sample_file):
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
            # The return value from the call(...) function is the readout Sequential() model, with
            # training set to `True`.
            readout = tf.map_fn(lambda x: model(x, training=True), features, dtype=tf.float32)

        # Having called this on one set of features, we have an initialised readout.
        # We squeeze the tensor to ... TODO
        readout = tf.squeeze(readout)
        # This is the reverse of the normalisation applied in the parse function in
        # omet_tfrecord_utils.

        return graph, readout, data_set_itrtr, label


def run_predictions(graph, readout, data_set_itrtr, labels, checkpoint_id, checkpoint_dir,
                    normalise_pred=True):

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        # Load the weights from the checkpoint
        try:
            saver.restore(sess, checkpoint_dir + '/model.ckpt-' + str(checkpoint_id))
        except Exception as ex:
            print(ex)
            assert False
        # We are going to take a median of a number of predictions
        predictions = []
        # We run the model 50 times to predict delays based for the network represented by
        # the sample data set.
        for _ in range(50):
            sess.run(data_set_itrtr.initializer)
            # The `true_delay` value here is the original delay value from the sample data set,
            # against which we compare the median value of the predicted delay below.
            # TODO check why labels has to be passed in
            # Note that we need to pass back the median of the `pred_delay` and the true_delay
            # just so that we have two tensors of the same shape for graphing purposes.
            predicted, true_vals = sess.run([readout, labels])
            if normalise_pred:
                predicted = 0.54 * predicted + 0.37
            predictions.append(predicted)

        median_prediction = np.median(predictions, axis=0)

        if normalise_pred:
            true_vals = 0.54 * true_vals + 0.37

    mse = mean_squared_error(median_prediction, true_vals[0])
    r2 = r2_score(median_prediction, true_vals[0])
    return median_prediction, predictions, true_vals, mse, r2


def do_test_prediction(sample_file, checkpoint_dir, checkpoint_id=50000, normalise_pred=True):
    graph, readout, data_set_itrtr, labels = get_model_readout(sample_file)
    _, _, _, mse, r2 = run_predictions(graph, readout, data_set_itrtr, labels, checkpoint_id,
                                       checkpoint_dir,
                                       normalise_pred=True)

    return mse, r2

