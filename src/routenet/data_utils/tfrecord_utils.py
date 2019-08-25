# -*- coding: utf-8 -*-
# Copyright (c) 2019, Krzysztof Rusek [^1], Paul Almasan [^2]
#
# [^1]: AGH University of Science and Technology, Department of
#     communications, Krakow, Poland. Email: krusek\@agh.edu.pl
#
# [^2]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: almasan@ac.upc.edu


import glob
import itertools as it
import os
import random
import re
import tarfile

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf


def process_data(network_data_dir, te_split=0.8):
    """
    The entry point into the utilities for converting NED and routing definitions,
    and network simulation results, into TensorFlow data.

    For a given network, the source data files are in the `network_data_dir`. The files and
    their contents are described in the Data_Files_and_Formats.md file in the same directory as
    this code.

    The data in the source files is transformed into TensorFlow formats and written to
    'training and 'evalaution' directories, in a 'tfrecords' output directory within the
    `network_data_dir`, where the TensorFlow data files are named corresponding to the results data
    file from which the data was processed.

    :param network_data_dir: The source directory for the network data to be processed.
    :param te_split: The percentage of data that should be written to the training data set,
                    with the remainder written to an evaluation data set.
    :return: None
    """
    # The directory is assumed to have a trailing '/', so make sure it does. It does not
    # matter if there are two, it does matter if there is not one at all.
    network_data_dir = network_data_dir + '/'
    network_name = network_data_dir.split('/')[-1]
    if network_name == '':
        network_name = network_data_dir.split('/')[-2]

    ned_file_name = network_data_dir + 'Network_' + network_name + '.ned'

    # Get all of the tar.gz results files in the network data directory
    for results_file_name in glob.glob(network_data_dir + '/*.tar.gz'):
        # Construct the file name for the TF records
        tf_file_name = results_file_name.split('/')[-1].split('.')[0] + '.tfrecords'

        # Open the network results data tar.gz
        tar_file = tarfile.open(results_file_name, 'r:gz')

        tar_info = tar_file.next()
        if not tar_info.isdir():
            raise Exception('Tar file with wrong format')

        results_file = tar_file.extractfile(tar_info.name + '/simulationResults.txt')
        routing_file = tar_file.extractfile(tar_info.name + '/Routing.txt')

        tf.logging.info('Starting ', results_file)
        make_tfrecord(network_data_dir, tf_file_name, ned_file_name, routing_file, results_file)

    tfr_dir_name = network_data_dir + '/tfrecords'

    tfr_train_dir_name = tfr_dir_name + '/train/'
    tfr_eval_dir_name = tfr_dir_name + '/evaluate/'
    if not os.path.exists(tfr_train_dir_name):
        os.makedirs(tfr_train_dir_name)

    if not os.path.exists(tfr_eval_dir_name):
        os.makedirs(tfr_eval_dir_name)

    tfrecords = glob.glob(tfr_dir_name + '/*.tfrecords')
    training_split = len(tfrecords) * te_split
    train_samples = random.sample(tfrecords, int(training_split))
    evaluate_samples = list(set(tfrecords) - set(train_samples))

    for file in train_samples:
        file_name = file.split('/')[-1]
        os.rename(file, tfr_train_dir_name + file_name)

    for file in evaluate_samples:
        file_name = file.split('/')[-1]
        os.rename(file, tfr_eval_dir_name + file_name)


def make_tfrecord(data_dir_path, tf_file_name, ned_file_name, routing_file, results_file):
    connections, num_nodes, link_capacity_dict = ned2lists(ned_file_name)

    routing = load_routing(routing_file)
    paths, link_capacities = make_paths(routing, connections, link_capacity_dict)

    num_paths = len(paths)
    num_links = max(max(paths)) + 1

    link_indices, path_indices, sequ_indices = make_indices(paths)
    n_total = len(path_indices)  # TODO what is n_total of?

    tfrecords_dir = data_dir_path + '/tfrecords/'

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)

    tf_rcrd_wrtr = tf.python_io.TFRecordWriter(tfrecords_dir + tf_file_name)
    rslt_pos_gnrtr = ResultsPositionGenerator(num_nodes)

    for result_data in results_file:
        # TODO the decode is for byte data, but why do we need that?
        result_data = result_data.decode().split(',')
        write_tfrecord(result_data, rslt_pos_gnrtr, num_nodes, num_paths, link_capacities,
                       link_indices, path_indices, sequ_indices, num_links, n_total, tf_rcrd_wrtr)
    tf_rcrd_wrtr.close()


def write_tfrecord(result_data,
                   rslt_pos_gnrtr,
                   num_nodes,
                   num_paths,
                   link_capacities,
                   link_indices,
                   path_indices,
                   sequ_indices,
                   num_links,
                   n_total,
                   tf_rcrd_wrtr):

    traffic_bw_txs, delays, jitters = get_corresponding_values(rslt_pos_gnrtr, result_data,
                                                               num_nodes, num_paths)

    tf_record = \
        tf.train.Example(features=tf.train.Features(
            feature={'traffic': _float_features(traffic_bw_txs),
                     'delay': _float_features(delays),
                     'jitter': _float_features(jitters),
                     'link_capacity': _float_features(link_capacities),
                     'links': _int64_features(link_indices),
                     'paths': _int64_features(path_indices),
                     'sequences': _int64_features(sequ_indices),
                     'n_links': _int64_feature(num_links),
                     'n_paths': _int64_feature(num_paths),
                     'n_total': _int64_feature(n_total)}))

    tf_rcrd_wrtr.write(tf_record.SerializeToString())


def ned2lists(ned_file_name):
    """
    Processes the NED file and scans for the lines in the "connections:" section.
    Those lines are of the format:

    node0.port[<port number>] <--> Channel<capacity>kbps <--> node2.port[<port number>];

    For example:

    node0.port[0] <--> Channel10kbps <--> node2.port[0];



    :param ned_file_name:
    :return:
    """
    channels = []
    link_capacity_dict = {}  # TODO change to link_capacity?
    with open(ned_file_name) as ned_file:
        p = re.compile(
            r'\s+node(\d+).port\[(\d+)\]\s+<-->\s+Channel(\d+)kbps+\s<-->\s+node(\d+).port\[(\d+)\]')
        for line in ned_file:
            connection_line = p.match(line)
            if connection_line:
                # For a connection_line: "node0.port[0] <--> Channel10kbps <--> node1.port[0]"
                # aux list will be [0, 0, 1, 0], which is to say [node0, port0, node1, port 0]
                node_port_list = []
                elem_cntr = 0
                for elem in list(map(int, connection_line.groups())):
                    if elem_cntr != 2:  # Skip the link capacity
                        node_port_list.append(elem)
                    elem_cntr = elem_cntr + 1
                channels.append(node_port_list)
                # The key in the link_capacity_dict is, based on the example above, 0:1, as in
                # node0:node1, and the value is 10, as in Channel10kbps.
                link_capacity_dict[
                    (connection_line.groups()[0]) + ':' + str(connection_line.groups()[3])] = int(
                    connection_line.groups()[2])

    # Find the largest node number in the channels, and add one to get the number of nodes
    num_nodes = max(map(max, channels)) + 1
    connections = [{} for i in range(num_nodes)]
    # Shape of connections[node][port] = node connected to
    for node_port in channels:
        # Based on the example above, connection from node0.port0 to node1
        connections[node_port[0]][node_port[1]] = node_port[2]
        # and connection from node1.port0 to node0
        connections[node_port[2]][node_port[3]] = node_port[0]
    # Connections store an array of nodes where each node position correspond to
    # another array of nodes that are connected to the current node
    connections = [[v for k, v in sorted(con.items())] for con in connections]
    return connections, num_nodes, link_capacity_dict


def load_routing(routing_file):
    """
    Reads the routing matrix from the routing file and returns a Numpy representation of the matrix.
    :param routing_file: File containing routing matrix. See Data_Files_and_Formats.md.
    :return: A Numpy representation of the routing matrix.
    """
    routing_df = pd.read_csv(routing_file, header=None, index_col=False)
    routing_df = routing_df.drop([routing_df.shape[0]], axis=1)
    # TODO use DataFrame.to_numpy()
    return routing_df.values


def make_paths(routing, connections, link_capacity_dict):
    num_nodes = routing.shape[0]
    edges, link_capacities = extract_links(num_nodes, connections, link_capacity_dict)
    paths = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                paths.append(
                    [edges.index(tup) for tup in pairwise(gen_path(routing, i, j, connections))])
    return paths, link_capacities


def extract_links(num_nodes, connections, link_capacity_dict):
    """

    :param num_nodes:
    :param connections:
    :param link_capacity_dict:
    :return:
    """
    A = np.zeros((num_nodes, num_nodes))

    for a, c in zip(A, connections):
        a[c] = 1

    graph = nx.from_numpy_array(A, create_using=nx.DiGraph())
    edges = list(graph.edges)
    link_capacities = []
    # The edges 0-2 or 2-0 can exist. They are duplicated (up and down) and they must have same
    # capacity.
    for edge in edges:
        if str(edge[0]) + ':' + str(edge[1]) in link_capacity_dict:
            capacity = link_capacity_dict[str(edge[0]) + ':' + str(edge[1])]
            link_capacities.append(capacity)
        elif str(edge[1]) + ':' + str(edge[0]) in link_capacity_dict:
            capacity = link_capacity_dict[str(edge[1]) + ':' + str(edge[0])]
            link_capacities.append(capacity)
        else:
            raise Exception('Error in dataset - edge not found - ', edge)

    return edges, link_capacities


def get_corresponding_values(rslt_pos_gnrtr, result_data, num_nodes, num_paths):
    traffic_bw_txs = np.zeros(num_paths)
    delays = np.zeros(num_paths)
    jitters = np.zeros(num_paths)
    itrtr = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                delay_pos = rslt_pos_gnrtr.get_delay_pos(i, j)
                jitter_pos = rslt_pos_gnrtr.get_jitter_pos(i, j)
                traffic_bw_tx_pos = rslt_pos_gnrtr.get_bw_pos(i, j)
                traffic_bw_txs[itrtr] = float(result_data[traffic_bw_tx_pos])
                delays[itrtr] = float(result_data[delay_pos])
                jitters[itrtr] = float(result_data[jitter_pos])
                itrtr = itrtr + 1

    return traffic_bw_txs, delays, jitters


def data(parsed_args):
    directory = parsed_args.d[0]
    process_data(directory)


def gen_path(routing, s, d, connections):
    while s != d:
        yield s
        s = connections[s][routing[s, d]]
    yield s


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


def make_indices(paths):
    link_indices = []
    path_indices = []
    sequ_indices = []
    segment = 0
    for p in paths:
        link_indices += p
        path_indices += len(p) * [segment]
        sequ_indices += list(range(len(p)))
        segment += 1
    return link_indices, path_indices, sequ_indices


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class ResultsPositionGenerator:
    _num_nodes = 0
    _offset_delay = 0
    has_packet_gen = True

    def __init__(self, num_nodes):
        self._num_nodes = num_nodes
        self._offset_delay = num_nodes * num_nodes * 3

    def get_bw_pos(self, src, dst):
        return (src * self._num_nodes + dst) * 3

    def get_gen_pckt_pos(self, src, dst):
        return (src * self._num_nodes + dst) * 3 + 1

    def get_drop_pckt_pos(self, src, dst):
        return (src * self._num_nodes + dst) * 3 + 2

    def get_delay_pos(self, src, dst):
        return self._offset_delay + (src * self._num_nodes + dst) * 7

    def get_jitter_pos(self, src, dst):
        return self._offset_delay + (src * self._num_nodes + dst) * 7 + 6


def parse(serialized, target='delay'):
    """
    Target is the name of predicted variable
    """

    # TODO 'traffic' below is bandwidth of traffic transmitted
    with tf.device('/cpu:0'):
        with tf.name_scope('parse'):
            features = tf.io.parse_single_example(serialized,
                                                  features={'traffic': tf.VarLenFeature(tf.float32),
                                                            target: tf.VarLenFeature(tf.float32),
                                                            'link_capacity': tf.VarLenFeature(
                                                                tf.float32),
                                                            'links': tf.VarLenFeature(tf.int64),
                                                            'paths': tf.VarLenFeature(tf.int64),
                                                            'sequences': tf.VarLenFeature(tf.int64),
                                                            'n_links': tf.FixedLenFeature([],
                                                                                          tf.int64),
                                                            'n_paths': tf.FixedLenFeature([],
                                                                                          tf.int64),
                                                            'n_total': tf.FixedLenFeature([],
                                                                                          tf.int64)})

            for k in ['traffic', target, 'link_capacity', 'links', 'paths', 'sequences']:
                # TODO why is this being applied?
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
