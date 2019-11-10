#@title Licensed under the MIT License (the "License"); { display-mode:"form" }
# MIT License

__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'
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


def get_sample(network_name, omnet_data_dir=None):
    random.seed(13)
    if not omnet_data_dir:
        omnet_data_dir = os.getenv('OMNET_DATA_DIR')
    train_data_path = omnet_data_dir + '/datasets_v0/' + network_name + '/tfrecords/evaluate/'
    train_data_filename = random.choice(os.listdir(train_data_path))
    sample_file = train_data_path + train_data_filename
    return sample_file


def get_plot_sample(pred_data, labels, sample_size):
    # Given the large number of delay predictions in a single sample, randomly sample some of the
    # results
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

    plt.bar(index, preds, bar_width, color='b', label='Predicted Delay')

    plt.bar(index + bar_width, labels, bar_width, color='greenyellow', label='True Delay')

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
    plt.xlabel('Relative error (true-median predicted)/true', fontsize=25)
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    plt.grid(color='gray', linestyle='-', linewidth=2, alpha=0.3)
    plt.title('CDF of the Mean Relative Error', fontsize=25)
    fig = plt.gcf()
    fig.set_size_inches(14, 8.5)
