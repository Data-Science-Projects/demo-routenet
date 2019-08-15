# -*- coding: utf-8 -*-
# Copyright (c) 2019, Krzysztof Rusek [^1], Paul Almasan [^2]
#
# [^1]: AGH University of Science and Technology, Department of
#     communications, Krakow, Poland. Email: krusek\@agh.edu.pl
#
# [^2]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: almasan@ac.upc.edu

import tensorflow as tf


def parse(serialized, target='delay'):
    """
    Target is the name of predicted variable
    """
    with tf.device('/cpu:0'):
        with tf.name_scope('parse'):
            features = \
                tf.io.parse_single_example(serialized,
                                           features={'traffic': tf.VarLenFeature(tf.float32),
                                                     target: tf.VarLenFeature(tf.float32),
                                                     'link_capacity': tf.VarLenFeature(tf.float32),
                                                     'links': tf.VarLenFeature(tf.int64),
                                                     'paths': tf.VarLenFeature(tf.int64),
                                                     'sequences': tf.VarLenFeature(tf.int64),
                                                     'n_links': tf.FixedLenFeature([], tf.int64),
                                                     'n_paths': tf.FixedLenFeature([], tf.int64),
                                                     'n_total': tf.FixedLenFeature([], tf.int64)})

            for k in ['traffic', target, 'link_capacity', 'links', 'paths', 'sequences']:
                features[k] = tf.sparse_tensor_to_dense(features[k])
                if k == 'delay':
                    features[k] = (features[k] - 0.37) / 0.54
                if k == 'traffic':
                    features[k] = (features[k] - 0.17) / 0.13
                if k == 'link_capacity':
                    features[k] = (features[k] - 25.0) / 40.0

    return {k: v for k, v in features.items() if k is not target}, features[target]


def read_dataset(filename, target='delay'):
    ds = tf.data.TFRecordDataset(filename)
    ds = ds.map(lambda buf: parse(buf, target=target))
    ds = ds.batch(1)
    it = ds.make_initializable_iterator()

    return it
