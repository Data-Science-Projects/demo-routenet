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

from routenet.new_parser import NewParser


def extract_links(n, connections, link_cap):
    A = np.zeros((n, n))

    for a, c in zip(A, connections):
        a[c] = 1

    graph = nx.from_numpy_array(A, create_using=nx.DiGraph())
    edges = list(graph.edges)
    capacities_links = []
    # The edges 0-2 or 2-0 can exist. They are duplicated (up and down) and they must have same
    # capacity.
    for edge in edges:
        if str(edge[0])+':'+str(edge[1]) in link_cap:
            capacity = link_cap[str(edge[0])+':'+str(edge[1])]
            capacities_links.append(capacity)
        elif str(edge[1])+':'+str(edge[0]) in link_cap:
            capacity = link_cap[str(edge[1])+':'+str(edge[0])]
            capacities_links.append(capacity)
        else:
            print("ERROR IN THE DATASET!")
            exit()
    return edges, capacities_links


def make_paths(routing, connections, link_cap):
    n = routing.shape[0]
    edges, capacities_links = extract_links(n, connections, link_cap)
    paths = []
    for i in range(n):
        for j in range(n):
            if i != j:
                paths.append([edges.index(tup) for tup in pairwise(gen_path(routing, i, j,
                                                                            connections))])
    return paths, capacities_links


def ned2lists(fname):
    channels = []
    link_cap = {}
    with open(fname) as f:
        p = re.compile(r'\s+node(\d+).port\[(\d+)\]\s+<-->\s+Channel(\d+)kbps+\s<-->\s+node(\d+).port\[(\d+)\]')
        for line in f:
            m = p.match(line)
            if m:
                aux_list = []
                elem_cntr = 0
                for elem in list(map(int, m.groups())):
                    if elem_cntr != 2:
                        aux_list.append(elem)
                    elem_cntr = elem_cntr + 1
                channels.append(aux_list)
                link_cap[(m.groups()[0])+':'+str(m.groups()[3])] = int(m.groups()[2])

    n = max(map(max, channels))+1
    connections = [{} for i in range(n)]
    # Shape of connections[node][port] = node connected to
    for c in channels:
        connections[c[0]][c[1]] = c[2]
        connections[c[2]][c[3]] = c[0]
    # Connections store an array of nodes where each node position correspond to
    # another array of nodes that are connected to the current node
    connections = [[v for k, v in sorted(con.items())]
                   for con in connections]
    return connections, n, link_cap


def get_corresponding_values(pos_parser, line, n, bws, delays, jitters):
    bws.fill(0)
    delays.fill(0)
    jitters.fill(0)
    itrtr = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                delay = pos_parser.get_delay_ptr(i, j)
                jitter = pos_parser.get_jitter_ptr(i, j)
                traffic = pos_parser.get_bw_ptr(i, j)
                bws[itrtr] = float(line[traffic])
                delays[itrtr] = float(line[delay])
                jitters[itrtr] = float(line[jitter])
                itrtr = itrtr + 1


def make_tfrecord2(data_dir_path, tf_file, ned_file, routing_file, data_file):
    con, n, link_cap = ned2lists(ned_file)
    pos_parser = NewParser(n)

    routing = load_routing(routing_file)
    paths, link_capacities = make_paths(routing, con, link_cap)

    n_paths = len(paths)
    n_links = max(max(paths)) + 1
    a = np.zeros(n_paths)
    d = np.zeros(n_paths)
    j = np.zeros(n_paths)

    tfrecords_dir = data_dir_path+'tfrecords/'

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)

    link_indices, path_indices, sequ_indices = make_indices(paths)
    n_total = len(path_indices)

    writer = tf.python_io.TFRecordWriter(tfrecords_dir + tf_file)

    for line in data_file:
        line = line.decode().split(',')
        get_corresponding_values(pos_parser, line, n, a, d, j)

        example = tf.train.Example(features=tf.train.Features(feature={
            'traffic': _float_features(a),
            'delay': _float_features(d),
            'jitter': _float_features(j),
            'link_capacity': _float_features(link_capacities),
            'links': _int64_features(link_indices),
            'paths': _int64_features(path_indices),
            'sequences': _int64_features(sequ_indices),
            'n_links': _int64_feature(n_links),
            'n_paths': _int64_feature(n_paths),
            'n_total': _int64_feature(n_total)
        }))

        writer.write(example.SerializeToString())
    writer.close()


def data(parsed_args):
    directory = parsed_args.d[0]
    process_data(directory)


def process_data(data_dir_path):
    # The directory is assumed to have a trailing '/', so make sure it does. It does not
    # matter if there are two, it does matter if there is not one at all.
    data_dir_path = data_dir_path + '/'
    nodes_dir = data_dir_path.split('/')[-1]
    if nodes_dir == '':
        nodes_dir = data_dir_path.split('/')[-2]

    ned_file = data_dir_path+'Network_'+nodes_dir+'.ned'

    for filename in os.listdir(data_dir_path):
        if filename.endswith('.tar.gz'):
            print(filename)
            tf_file = filename.split('.')[0]+'.tfrecords'
            tar = tarfile.open(data_dir_path+filename, 'r:gz')

            dir_info = tar.next()
            if not dir_info.isdir():
                raise Exception('Tar file with wrong format')

            delay_file = tar.extractfile(dir_info.name + '/simulationResults.txt')
            routing_file = tar.extractfile(dir_info.name + '/Routing.txt')

            tf.logging.info('Starting ', delay_file)
            make_tfrecord2(data_dir_path, tf_file, ned_file, routing_file, delay_file)

    directory_tfr = data_dir_path+'tfrecords/'

    tfr_train = directory_tfr+'train/'
    tfr_eval = directory_tfr+'evaluate/'
    if not os.path.exists(tfr_train):
        os.makedirs(tfr_train)

    if not os.path.exists(tfr_eval):
        os.makedirs(tfr_eval)

    tfrecords = glob.glob(directory_tfr + '*.tfrecords')
    training = len(tfrecords) * 0.8
    train_samples = random.sample(tfrecords, int(training))
    evaluate_samples = list(set(tfrecords) - set(train_samples))

    for file in train_samples:
        file_name = file.split('/')[-1]
        os.rename(file, tfr_train + file_name)

    for file in evaluate_samples:
        file_name = file.split('/')[-1]
        os.rename(file, tfr_eval + file_name)


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


def load_routing(routing_file):
    routing_df = pd.read_csv(routing_file, header=None, index_col=False)
    routing_df = routing_df.drop([routing_df.shape[0]], axis=1)
    return routing_df.values


def make_indices(paths):
    link_indices = []
    path_indices = []
    sequ_indices = []
    segment = 0
    for p in paths:
        link_indices += p
        path_indices += len(p)*[segment]
        sequ_indices += list(range(len(p)))
        segment += 1
    return link_indices, path_indices, sequ_indices


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
