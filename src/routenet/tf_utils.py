# -*- coding: utf-8 -*-
# Copyright (c) 2019, Krzysztof Rusek [^1], Paul Almasan [^2]
#
# [^1]: AGH University of Science and Technology, Department of
#     communications, Krakow, Poland. Email: krusek\@agh.edu.pl
#
# [^2]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: almasan@ac.upc.edu


import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
