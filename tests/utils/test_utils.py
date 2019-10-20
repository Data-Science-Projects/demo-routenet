# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

import tensorflow as tf

from routenet.model.routenet_model import RouteNetModel
from routenet.train.train import read_dataset


# TODO This code should be refactored with the code in the rn_notebook_utils.py


def get_model_readout(test_sample_file, normalise_readout=False):
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
