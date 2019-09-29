#@title Licensed under the MIT License (the "License"); { display-mode:"form" }
# MIT License

# Copyright (c) 2019 Paul Almasan, José Suárez-Varela, Krzysztzof Rusek

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
The code in this file was based on the code from the original demo notebook here:
https://github.com/knowledgedefinednetworking/demo-routenet/blob/master/demo_notebooks/demo.ipynb
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from routenet.data_utils.omnet_tfrecord_utils import read_dataset
from routenet.model.routenet_model import RouteNetModel


def get_sample(network_name):
    random.seed(13)
    # Path to data sets
    omnet_data_dir = os.getenv('OMNET_DATA_DIR')
    train_data_path = omnet_data_dir+'/' + network_name + '/tfrecords/train/'
    train_data_filename = random.choice(os.listdir(train_data_path))
    sample_file = train_data_path + train_data_filename
    # print(sample_file.split('/')[-1])
    return sample_file


def get_model_readout(test_sample_file):
    graph = tf.Graph()
    with graph.as_default():
        model = RouteNetModel()
        model.build()

        data_set = read_dataset(test_sample_file)
        data_set_itrtr = data_set.make_initializable_iterator()
        # The `label` here is the delay value associated with the features. The features are selected in
        # the transformation_func(...) from the train module.
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
        # TODO the rationale for the normalisation has to be explained.
        readout = 0.54 * readout + 0.37

        return graph, readout, data_set_itrtr, label


def run_predictions(graph, readout, data_itrtr, true_value):
    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        # Load the weights from the checkpoint
        saver.restore(sess, '../model_checkpoints-imac/model.ckpt-8151')

        # We are going to take a median of a number of predictions
        predictions = []
        # We run the model 50 times to predict delays based for the network represented by the sample
        # data set.
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

        return median_prediction, predictions, true_val


def get_plot_sample(pred_data, labels, sample_size):
    # Given the large number of delay predictions in a single sample, randomly sample some of the results
    ids = random.sample(range(0, len(pred_data)), sample_size)
    ids.sort()

    sample_predictions = []
    sample_labels = []

    for i in ids:
        sample_predictions.append(pred_data[i])
        sample_labels.append(labels[0][i])

    return sample_predictions, sample_labels


def plot_pred_vs_true_bar(preds, labels, sample_size):

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(sample_size)
    bar_width = 0.40

    rects1 = plt.bar(index, preds, bar_width, color='b', label='Predicted Delay')

    rects2 = plt.bar(index + bar_width, labels, bar_width, color='greenyellow', label='True Delay')

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)

    plt.xlabel('Paths', fontsize=25)
    plt.ylabel('Delay', fontsize=25)
    plt.title('Median Predicted Delay vs True Delay', fontsize=25)
    plt.legend(loc='center', fontsize=15, bbox_to_anchor=(0.5, -0.20), fancybox=True, ncol=2)
    plt.tight_layout()

    fig = plt.gcf()
    fig.set_size_inches(14, 8.5)


def plot_pred_vs_true_scatter(median, all_preds, labels):
    ax = plt.subplot()

    xerr = [median - np.percentile(all_preds, q=5, axis=0),
            np.percentile(all_preds, q=95, axis=0) - median]

    ax.errorbar(x=median, y=labels[0], fmt='o', xerr=xerr, alpha=0.8, ecolor='silver')

    m = max(labels[0])
    ax.plot([0, 1.3 * m], [0, 1.3 * m], 'k')
    ax.grid(color='gray', linestyle='-', linewidth=2, alpha=0.3)
    ax.set_xlabel('Prediction', fontsize=25)
    ax.set_ylabel('True Delay', fontsize=25)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)
    ax.set_xlim(left=-0.002, right=1.2 * m)
    ax.set_ylim(bottom=-0.005, top=1.2 * m)
    fig = plt.gcf()
    plt.tight_layout()
    fig.set_size_inches(14, 8.5)


def plot_cdf(labels, median):
    def frange(x, y, jump):
        while x < y:
            yield x
            x += jump

    mre = (labels - median) / labels
    mre = np.sort(mre)

    mre = np.insert(mre, 0, -15.0)
    mre = np.append(mre, 15.0)

    plt.hist(mre, cumulative=True,
             histtype='step',
             bins=10000,
             alpha=0.8,
             color='blue',
             density=True,
             linewidth=3)

    plt.ylim((-0.005, 1.005))
    plt.xlim((mre[1], mre[-2]))
    plt.xlabel("Relative error (true-median predicted)/true", fontsize=25)
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    plt.grid(color='gray', linestyle='-', linewidth=2, alpha=0.3)
    plt.title('CDF of the Mean Relative Error',fontsize=25)
    fig = plt.gcf()
    fig.set_size_inches(14, 8.5)
