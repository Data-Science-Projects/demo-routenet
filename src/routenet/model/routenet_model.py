# -*- coding: utf-8 -*-
# Copyright (c) 2019, Krzysztof Rusek [^1], Paul Almasan [^2]
#
# [^1]: AGH University of Science and Technology, Department of
#     communications, Krakow, Poland. Email: krusek\@agh.edu.pl
#
# [^2]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: almasan@ac.upc.edu


import sys

import tensorflow as tf
from tensorflow import keras


class RouteNetModel(tf.keras.Model):
    # TODO These parameters are NOT the same as the original code, but are the same as those
    # used in the demo notebook. It is not clear which parameters were used elsewhere in the
    # original data pipeline.

    default_hparams = tf.contrib.training.HParams(link_state_dim=32,
                                                  path_state_dim=32,
                                                  T=8,
                                                  readout_units=256,
                                                  learning_rate=0.001,
                                                  batch_size=32,
                                                  dropout_rate=0.5,
                                                  l2=0.1,
                                                  l2_2=0.01,
                                                  learn_embedding=True)
    # If false, only the readout is trained

    def __init__(self, hparams=default_hparams, output_units=1, final_activation=None):
        super(RouteNetModel, self).__init__()
        self.hparams = hparams

        self.edge_update = tf.keras.layers.GRUCell(self.hparams.link_state_dim)
        self.path_update = tf.keras.layers.GRUCell(self.hparams.path_state_dim)

        self.readout = tf.keras.models.Sequential()

        self.readout.add(keras.layers.Dense(hparams.readout_units,
                                            activation=tf.nn.selu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.hparams.l2)))
        self.readout.add(keras.layers.Dropout(rate=hparams.dropout_rate))
        self.readout.add(keras.layers.Dense(hparams.readout_units,
                                            activation=tf.nn.selu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.hparams.l2)))
        self.readout.add(keras.layers.Dropout(rate=hparams.dropout_rate))

        self.readout.add(keras.layers.Dense(output_units,
                                            kernel_regularizer=
                                            tf.contrib.layers.l2_regularizer(self.hparams.l2_2),
                                            activation=final_activation))

    def build(self, input_shape=None):
        del input_shape
        self.edge_update.build(tf.TensorShape([None, self.hparams.path_state_dim]))
        self.path_update.build(tf.TensorShape([None, self.hparams.link_state_dim]))
        self.readout.build(input_shape=[None, self.hparams.path_state_dim])
        self.built = True

    def call(self, inputs, training=False):
        print('******** in call ******************', file=sys.stderr)

        f_ = inputs
        shape = tf.stack([f_['n_links'], self.hparams.link_state_dim - 1], axis=0)
        link_state = tf.concat([
            tf.expand_dims(f_['link_capacity'], axis=1),
            tf.zeros(shape)
        ], axis=1)
        shape = tf.stack([f_['n_paths'], self.hparams.path_state_dim - 1], axis=0)
        path_state = tf.concat([
            tf.expand_dims(f_['traffic'][0:f_['n_paths']], axis=1),
            tf.zeros(shape)
        ], axis=1)

        links = f_['links']
        paths = f_['paths']
        seqs = f_['sequences']

        for _ in range(self.hparams.T):
            h_tild = tf.gather(link_state, links)

            ids = tf.stack([paths, seqs], axis=1)
            max_len = tf.reduce_max(seqs)+1
            shape = tf.stack([f_['n_paths'], max_len, self.hparams.link_state_dim])
            lens = tf.math.segment_sum(data=tf.ones_like(paths),
                                       segment_ids=paths)

            link_inputs = tf.scatter_nd(ids, h_tild, shape)
            outputs, path_state = tf.nn.dynamic_rnn(self.path_update,
                                                    link_inputs,
                                                    sequence_length=lens,
                                                    initial_state=path_state,
                                                    dtype=tf.float32)
            m = tf.gather_nd(outputs, ids)
            m = tf.math.unsorted_segment_sum(m, links, f_['n_links'])

            # Keras cell expects a list
            link_state, _ = self.edge_update(m, [link_state])

        if self.hparams.learn_embedding:
            readout = self.readout(path_state, training=training)
        else:
            readout = self.readout(tf.stop_gradient(path_state), training=training)

        return readout
