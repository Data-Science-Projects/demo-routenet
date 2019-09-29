# -*- coding: utf-8 -*-
# Copyright (c) 2019, Krzysztof Rusek [^1], Paul Almasan [^2]
#
# [^1]: AGH University of Science and Technology, Department of
#     communications, Krakow, Poland. Email: krusek\@agh.edu.pl
#
# [^2]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: almasan@ac.upc.edu

"""
Utilities to transform the data produced by the OMNeT++ network simulator into TF Records.
TODO These functions should be refactored in to a class so as to use member variables instead
of passing around data from one funtion to another.
"""

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
    their contents are described in the OMNet_Data_Files_and_Formats.md file in the same directory as
    this code.

    The data in the source files is transformed into TensorFlow formats and written to
    'training and 'evalaution' directories, in a 'tfrecords' output directory within the
    `network_data_dir`, where the TensorFlow data files are named corresponding to the results data
    file from which the data was processed.

    :param network_data_dir: The source directory for the network data to be processed, and the
    directory within which the TF records are created.
    :param te_split: The percentage of data that should be written to the training data set,
                    with the remainder written to an evaluation data set.
    :return: None
    """
    # The directory is assumed to have a trailing '/', so make sure it does.
    # TODO change to an assert and throw exception.
    if network_data_dir[-1] != '/':
        network_data_dir = network_data_dir + '/'
    network_name = network_data_dir.split('/')[-2]

    ned_file_name = network_data_dir + 'Network_' + network_name + '.ned'
    connections_lists, num_nodes, link_capacity_dict = ned2lists(ned_file_name)

    # Get all of the tar.gz results files in the network data directory
    for results_bundle_file_name in glob.glob(network_data_dir + '/*.tar.gz'):
        # Construct the file name for the TF records
        tf_file_name = results_bundle_file_name.split('/')[-1].split('.')[0] + '.tfrecords'

        # Open the network results data tar.gz
        tar_file = tarfile.open(results_bundle_file_name, 'r:gz')

        tar_info = tar_file.next()
        if not tar_info.isdir():
            raise Exception('Tar file with wrong format')

        results_file = tar_file.extractfile(tar_info.name + '/simulationResults.txt')
        routing_file = tar_file.extractfile(tar_info.name + '/Routing.txt')
        routing_mtrx = get_routing_matrix(routing_file)

        tf.logging.info('Starting ', results_file)
        make_tfrecord(network_data_dir,
                      tf_file_name,
                      connections_lists,
                      num_nodes,
                      link_capacity_dict,
                      routing_mtrx,
                      results_file)

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


def make_tfrecord(network_data_dir,
                  tf_file_name,
                  connections_lists,
                  num_nodes,
                  link_capacity_dict,
                  routing_mtrx,
                  results_file):
    """
    This function iterataes over the data in the `results_file` and uses the write_tfrecord(...)
    to write TF Records to the `tf_file_name` in the `network_data_dir`.

    :param network_data_dir: The directory into which the TF Record files are written.
    :param tf_file_name: The name of the TF Records file wriiten.
    :param connections_lists: See ned2lists(...)
    :param num_nodes: The number of nodes in the topology.
    :param link_capacity_dict: See ned2lists(...)
    :param routing_mtrx: See get_routing_matrix.
    :param results_file: The results file from which the source data is taken, and written
    by the write_tfrecord(...) function.
    """

    paths, link_capacities = make_paths(routing_mtrx, connections_lists, link_capacity_dict)

    num_paths = len(paths)
    num_links = max(max(paths)) + 1

    link_indices, path_indices, sequ_indices = make_indices(paths)
    n_total = len(path_indices)  # TODO what is n_total of?

    tfrecords_dir = network_data_dir + '/tfrecords/'

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)

    tf_rcrd_wrtr = tf.python_io.TFRecordWriter(tfrecords_dir + tf_file_name)
    rslt_pos_gnrtr = ResultsPositionGenerator(num_nodes)

    for result_data in results_file:
        # TODO the decode is for byte data, but why do we need that?
        result_data = result_data.decode().split(',')
        write_tfrecord(result_data, rslt_pos_gnrtr, num_paths, link_capacities,
                       link_indices, path_indices, sequ_indices, num_links, n_total, tf_rcrd_wrtr)
    tf_rcrd_wrtr.close()


def write_tfrecord(result_data,
                   rslt_pos_gnrtr,
                   num_paths,
                   link_capacities,
                   link_indices,
                   path_indices,
                   sequ_indices,
                   num_links,
                   n_total,
                   tf_rcrd_wrtr):

    traffic_bw_txs, delays, jitters = get_corresponding_values(rslt_pos_gnrtr, result_data,
                                                               num_paths)

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
    Processes the NED file and scans for the lines in the "connections_dicts:" section.
    Those lines are of the format:

    node0.port[<port number>] <--> Channel<capacity>kbps <--> node2.port[<port number>];

    For example:

    node0.port[0] <--> Channel10kbps <--> node1.port[0];
    node0.port[0] <--> Channel10kbps <--> node2.port[0];
    ...

    The return value of connections_lists is a list of lists where each connections_lists position
    corresponds to a list of other nodes connected to the node at that position in the
    connections_lists, for example [[1, 3, 2], [0, 2, 7], ...] means that node0 is connected to
    nodes 1, 3 and 2, and that node 1 is connected to nodes 0, 2 and 7. The order of the nodes in
    the sub-lists is defined by the order in which the `connections_lists:` in the .ned file appears,
    which is in the order of the port number, so, for example, node0 is connected to node1 via
    port0, which is shown by `node0.port[0] <--> Channel10kbps <--> node1.port[0];` in the .ned
    file.

    The return value of link_capacity_dict is a dictionary keyed by <noden>:<nodem>, where the
    value is the capacity between noden and nodem. For example, from node0 to node1 the capacity is
    10, so the dictionary entry is {'0:1':10}.

    :param ned_file_name:
    :return: connections_lists, described above; num_nodes, which is the number of nodes in the
    topology; and link_capacity_dict, described above.
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
                    (connection_line.groups()[0]) + ':' + str(connection_line.groups()[3])] = \
                    int(connection_line.groups()[2])

    # Find the largest node number in the channels, and add one to get the number of nodes
    num_nodes = max(map(max, channels)) + 1
    connections_dicts = [{} for i in range(num_nodes)]
    # Shape of connections_dicts[node][port] = node connected to
    for node_port in channels:
        # Based on the example above, connection from node0.port0 to node1
        connections_dicts[node_port[0]][node_port[1]] = node_port[2]
        # and connection from node1.port0 to node0
        connections_dicts[node_port[2]][node_port[3]] = node_port[0]
    connections_lists = [[v for k, v in sorted(con.items())] for con in connections_dicts]
    return connections_lists, num_nodes, link_capacity_dict


def get_routing_matrix(routing_file):
    """
    Reads the routing matrix from the routing file and returns a Numpy representation of the matrix.
    :param routing_file: File containing routing matrix. See OMNet_Data_Files_and_Formats.md.
    :return: A Numpy representation of the routing matrix.
    """
    routing_df = pd.read_csv(routing_file, header=None, index_col=False)
    routing_df = routing_df.drop([routing_df.shape[0]], axis=1)
    # TODO use DataFrame.to_numpy()
    return routing_df.values


def make_paths(routing_mtrx, connections_lists, link_capacity_dict):
    """
    This function calls `src_to_dest_hops(...)` to generate a list of tuples that are
    (src, dest) sequences. The function `extract_links(...)` generates the list of links.

    The paths list return value is populated with the index values of the tuples from
    `src_to_dest_hops` in links.

    For example, paths=[[0], [1], [2], [2, 10], [1, 8], ...] means that the path from node0 to node1
    is the first value, '[0]' in links, which is (0, 1), whilst the path from node0 to node4 is the
    third value in links, (0, 3) and the eleventh value, (3, 4). Examples based on nsfnetbw.

    :param routing_mtrx: A Numpy representation of the routing matrix.
    :param connections_lists: See ned2lists(...).
    :param link_capacity_dict: See extract_links(...)
    :return: paths, link_capacities
    """
    num_nodes = routing_mtrx.shape[0]
    links, link_capacities = extract_links(num_nodes, connections_lists, link_capacity_dict)
    paths = []
    for source in range(num_nodes):
        for destination in range(num_nodes):
            if source != destination:
                paths.append(
                    [links.index(tup) for tup in src_to_dest_hops(gen_path(routing_mtrx,
                                                                           source,
                                                                           destination,
                                                                           connections_lists))])
    return paths, link_capacities


def src_to_dest_hops(path_itrtr):
    """
    This function starts with the iterator from `gen_path`, then creates two iterators. The second
    iterator is incremented, and then both are zip'ed. The outcome is a list of tuples representing
    the source node and the next hop, then the next hop and its next hop and so on to the last next
    hop and destination, of the form:

    [(source, next hop 1), (next hop 1, next hop 2) ... (next hop n, destination)]

    :param path_itrtr: See gen_path(...)
    :return: A list of tuples representing the source node and the next hops from src to dest.
    """
    src_itrtr, next_hop_itrtr = it.tee(path_itrtr)
    next(next_hop_itrtr, None)
    return zip(src_itrtr, next_hop_itrtr)


def gen_path(routing_mtrx, source, destination, connections_lists):
    """
    Yields a sequence of node numbers in the path from the source to the destination,
    starting with the source and ending with the destination.

    :param routing_mtrx: A Numpy representation of the routing matrix.
    :param source: The source node number.
    :param destination: The destination node number.
    :param connections_lists: See ned2lists(...).
    :return: An iterator.
    """
    while source != destination:
        # The first value that is yielded is the source node number.
        yield source
        # The value of routing[source, destination] is the port number of the source for the path
        # from the source to the destination.
        # The value of connections_lists[source] is the list of other nodes to which the source node
        # has connections_lists, in order of port number.
        # Applying the port number as the index for the list of other nodes means that source
        # is then the value of one of the nodes to which source has a path, which is then yielded
        # also, up tp the point where the destination is encountered, when the while loop ends and
        # ...
        source = connections_lists[source][routing_mtrx[source, destination]]
        # ... the final value of source, which is the destination, is yielded. The result is the
        # sequence of node numbers in the path from the source to the destination, starting with
        # the source and ending with the destination.
    yield source


def extract_links(num_nodes, connections, link_capacity_dict):
    """

    :param num_nodes:
    :param connections:
    :param link_capacity_dict:
    :return:
    """
    # An adjacency matrix representation of a graph
    grph_adjcny_mtrx = np.zeros((num_nodes, num_nodes))

    #
    for adjacencies, connection_set in zip(grph_adjcny_mtrx, connections):
        # For adjacencies row n of the matrix, corresponding to node n, set a value of 1 for each
        # position in the row corresponding to the other nodes that node n has a link to.
        adjacencies[connection_set] = 1

    # Given the adjacency matrix, construct a directed graph.
    graph = nx.from_numpy_array(grph_adjcny_mtrx, create_using=nx.DiGraph())
    # From the graph, "Edges are represented as links between nodes ...". The links is a list
    # of tuples representing the connections_lists from node n to node m, and node m to node n.
    links = list(graph.edges)
    # The link_capacities is a list of the capacities of the links in order of the the connections_lists
    # in links.
    link_capacities = []
    # The links are duplicated from n to m and m to n, so they must have the same capacity in both
    # directions. The link_capacity_dict keys are of the form n:m and m:n.
    for link in links:
        if str(link[0]) + ':' + str(link[1]) in link_capacity_dict:
            capacity = link_capacity_dict[str(link[0]) + ':' + str(link[1])]
            link_capacities.append(capacity)
        elif str(link[1]) + ':' + str(link[0]) in link_capacity_dict:
            capacity = link_capacity_dict[str(link[1]) + ':' + str(link[0])]
            link_capacities.append(capacity)
        else:
            raise Exception('Error in dataset - link not found in link capacities - ', link)

    return links, link_capacities


def get_corresponding_values(rslt_pos_gnrtr, result_data, num_paths):
    traffic_bw_txs = np.zeros(num_paths)
    delays = np.zeros(num_paths)
    jitters = np.zeros(num_paths)
    itrtr = 0
    for i in range(rslt_pos_gnrtr.num_nodes):
        for j in range(rslt_pos_gnrtr.num_nodes):
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
    num_nodes = 0
    _offset_delay = 0
    has_packet_gen = True

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self._offset_delay = num_nodes * num_nodes * 3

    def get_bw_pos(self, src, dst):
        return (src * self.num_nodes + dst) * 3

    def get_gen_pckt_pos(self, src, dst):
        return (src * self.num_nodes + dst) * 3 + 1

    def get_drop_pckt_pos(self, src, dst):
        return (src * self.num_nodes + dst) * 3 + 2

    def get_delay_pos(self, src, dst):
        return self._offset_delay + (src * self.num_nodes + dst) * 7

    def get_jitter_pos(self, src, dst):
        return self._offset_delay + (src * self.num_nodes + dst) * 7 + 6


def parse(serialized, target='delay'):
    # Target is the name of predicted variable
    # TODO 'traffic' below is bandwidth of traffic transmitted
    with tf.device('/cpu:0'):
        with tf.name_scope('parse'):
            features = tf.io.parse_single_example(serialized,
                                                  features={'traffic': tf.io.VarLenFeature(tf.float32),
                                                            target: tf.io.VarLenFeature(tf.float32),
                                                            'link_capacity': tf.io.VarLenFeature(
                                                                tf.float32),
                                                            'links': tf.io.VarLenFeature(tf.int64),
                                                            'paths': tf.io.VarLenFeature(tf.int64),
                                                            'sequences': tf.io.VarLenFeature(tf.int64),
                                                            'n_links': tf.io.FixedLenFeature([],
                                                                                             tf.int64),
                                                            'n_paths': tf.io.FixedLenFeature([],
                                                                                             tf.int64),
                                                            'n_total': tf.io.FixedLenFeature([],
                                                                                             tf.int64)})

            for feature in ['traffic', target, 'link_capacity', 'links', 'paths', 'sequences']:
                # TODO This is a form of normalisation, but why these values? Factor into
                # discrete functions.
                features[feature] = tf.sparse.to_dense(features[feature])
                if feature == 'delay' and feature != target:
                    features[feature] = (features[feature] - 0.37) / 0.54
                if feature == 'traffic':
                    features[feature] = (features[feature] - 0.17) / 0.13
                if feature == 'link_capacity':
                    features[feature] = (features[feature] - 25.0) / 40.0

    return {k: v for k, v in features.items() if k is not target}, features[target]


def read_dataset(filename, target='delay'):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(lambda buf: parse(buf, target=target))
    dataset = dataset.batch(1)

    return dataset
