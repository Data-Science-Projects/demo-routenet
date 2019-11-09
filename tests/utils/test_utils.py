# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

from routenet.model.routenet_model_v0 import RouteNetModelV0
from routenet.train.normvals_v0 import NormVals
from routenet.train.train_v0 import read_dataset


def get_model_readout(test_sample_file):
    graph = tf.Graph()
    with graph.as_default():
        model = RouteNetModelV0()
        model.build()

        dataset = read_dataset(test_sample_file)
        # TODO Change to use dataset iteration with eager execution
        data_set_itrtr = tf.compat.v1.data.make_initializable_iterator(dataset)
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
                    norm_vals=NormVals()):

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
            predicted = norm_vals.std_delay * predicted + norm_vals.mean_delay
            predictions.append(predicted)

        median_prediction = np.median(predictions, axis=0)

        true_vals = norm_vals.std_delay * true_vals + norm_vals.mean_delay

    mse = mean_squared_error(median_prediction, true_vals[0])
    r2 = r2_score(median_prediction, true_vals[0])
    return median_prediction, predictions, true_vals, mse, r2


def do_test_prediction(sample_file, checkpoint_dir, checkpoint_id=50000, norm_vals=NormVals()):
    graph, readout, data_set_itrtr, labels = get_model_readout(sample_file)
    _, _, _, mse, r2 = run_predictions(graph,
                                       readout,
                                       data_set_itrtr,
                                       labels,
                                       checkpoint_id,
                                       checkpoint_dir,
                                       norm_vals=norm_vals)

    return mse, r2

