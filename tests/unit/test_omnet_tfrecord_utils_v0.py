# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

import os
import shutil
import unittest

import numpy as np
import tensorflow as tf

import routenet.data_utils.omnet_tfrecord_utils_v0 as tfr_utils
import routenet.train.train_v0 as rn_train

TEST_CODE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestTFRecords(unittest.TestCase):

    num_nodes = 0
    connections_lists = []
    link_capacity_dict = {}
    routing_mtrx = []
    paths = []
    link_capacities = []
    links = []
    traffic_bw_txs = np.zeros(1)
    delays = np.zeros(1)
    jitters = np.zeros(1)
    link_indices, path_indices, sequ_indices = [], [], []
    result_data = []
    rslt_pos_gnrtr = None
    network_data_dir = TEST_CODE_DIR + '/../unit-resources/nsfnetbw/'
    tf_rcrds_dir = network_data_dir + '/tfrecords/'
    tf_rcrds_fl_nm = tf_rcrds_dir + '/test_results.tfrecords'

    def test_a_ned2lists(self):
        ned_file_name = TEST_CODE_DIR + '/../unit-resources/nsfnetbw/Network_nsfnetbw.ned'
        self.__class__.connections_lists, self.__class__.num_nodes, \
            self.__class__.link_capacity_dict = tfr_utils.ned2lists(ned_file_name)
        assert (self.__class__.num_nodes == 14)
        assert (self.__class__.connections_lists == [[1, 3, 2], [0, 2, 7], [0, 1, 5], [0, 4, 8],
                                                     [3, 5, 6], [2, 4, 12, 13], [4, 7], [1, 6, 10],
                                                     [3, 9, 11], [8, 10, 12], [7, 9, 11, 13], [8, 10, 12],
                                                     [5, 9, 11], [5, 10]])

        assert (self.__class__.link_capacity_dict == {'0:1': 10, '0:3': 10, '0:2': 10, '1:2': 10,
                                                      '1:7': 10, '2:5': 10, '3:4': 40, '3:8': 10,
                                                      '4:5': 40, '4:6': 10, '5:12': 10, '5:13': 10,
                                                      '6:7': 10, '7:10': 40, '8:9': 10, '8:11': 10,
                                                      '9:10': 10, '9:12': 10, '10:11': 10,
                                                      '10:13': 10, '11:12': 10})

    def test_b_load_routing(self):
        routing_file = TEST_CODE_DIR + '/../unit-resources/nsfnetbw/Routing.txt'
        routing_expected = np.array([[-1, 0, 2, 1, 1, 2, 1, 0, 1, 1, 0, 1, 2, 2],
                                     [0, -1, 1, 0, 2, 1, 2, 2, 0, 2, 2, 2, 1, 1],
                                     [0, 1, -1, 0, 2, 2, 2, 1, 0, 2, 2, 2, 2, 2],
                                     [0, 0, 0, -1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1],
                                     [0, 2, 1, 0, -1, 1, 2, 2, 0, 1, 1, 0, 1, 1],
                                     [0, 0, 0, 1, 1, -1, 1, 1, 1, 2, 3, 2, 2, 3],
                                     [0, 1, 0, 0, 0, 0, -1, 1, 0, 1, 1, 1, 0, 0],
                                     [0, 0, 0, 1, 1, 1, 1, -1, 2, 2, 2, 2, 2, 2],
                                     [0, 0, 0, 0, 0, 2, 0, 1, -1, 1, 1, 2, 2, 1],
                                     [0, 1, 2, 0, 0, 2, 1, 1, 0, -1, 1, 0, 2, 1],
                                     [0, 0, 3, 1, 3, 3, 0, 0, 1, 1, -1, 2, 1, 3],
                                     [0, 1, 2, 0, 2, 2, 1, 1, 0, 2, 1, -1, 2, 1],
                                     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, -1, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, -1]])

        self.__class__.routing_mtrx = tfr_utils.get_routing_matrix(routing_file)
        np.testing.assert_array_equal(self.routing_mtrx, routing_expected)

    def test_c_make_paths(self):
        self.__class__.paths, self.__class__.link_capacities = \
            tfr_utils.make_paths(self.__class__.routing_mtrx,
                                 self.__class__.connections_lists,
                                 self.__class__.link_capacity_dict)

        paths_expected = [[0], [1], [2], [2, 10], [1, 8], [2, 10, 14], [0, 5], [2, 11], [2, 11, 25],
                          [0, 5, 23], [2, 11, 26], [1, 8, 17], [1, 8, 18], [3], [4], [3, 2],
                          [5, 22, 19], [4, 8], [5, 22], [5], [3, 2, 11], [5, 23, 31], [5, 23],
                          [5, 23, 32], [4, 8, 17], [4, 8, 18], [6], [7], [6, 2], [8, 16], [8],
                          [8, 16, 14], [7, 5], [6, 2, 11], [8, 17, 38], [8, 18, 41], [8, 17, 39],
                          [8, 17], [8, 18], [9], [9, 0], [9, 1], [10], [10, 13], [10, 14],
                          [10, 14, 20], [11], [11, 25], [11, 25, 28], [11, 26], [11, 26, 36],
                          [10, 13, 18], [12, 9], [14, 20, 21], [13, 15], [12], [13], [14], [14, 20],
                          [12, 11], [13, 17, 38], [13, 18, 41], [12, 11, 26], [13, 17], [13, 18],
                          [15, 6], [15, 7], [15], [16, 12], [16], [16, 14], [16, 14, 20],
                          [16, 12, 11], [17, 38], [18, 41], [17, 39], [17], [18], [19, 12, 9],
                          [20, 21], [19, 13, 15], [19, 12], [19], [19, 13], [20], [19, 12, 11],
                          [20, 23, 31], [20, 23], [20, 23, 32], [19, 13, 17], [19, 13, 18], [21, 3],
                          [21], [21, 4], [22, 19, 12], [22, 19], [22, 19, 13], [22], [23, 31, 27],
                          [23, 31], [23], [23, 32], [23, 31, 29], [23, 33], [24, 9], [24, 9, 0],
                          [24, 9, 1], [24], [24, 10], [26, 36, 37], [24, 10, 14], [25, 28, 30],
                          [25], [25, 28], [26], [26, 36], [25, 28, 33], [27, 24, 9], [28, 30, 21],
                          [29, 37, 15], [27, 24], [27, 24, 10], [29, 37], [28, 30, 22], [28, 30],
                          [27], [28], [27, 26], [29], [28, 33], [30, 21, 3], [30, 21], [33, 40, 15],
                          [31, 27, 24], [33, 40, 16], [33, 40], [30, 22], [30], [31, 27], [31],
                          [32], [31, 29], [33], [34, 24, 9], [35, 30, 21], [36, 37, 15], [34, 24],
                          [36, 37, 16], [36, 37], [35, 30, 22], [35, 30], [34], [36, 38], [35],
                          [36], [35, 33], [37, 15, 6], [37, 15, 7], [37, 15], [37, 16, 12],
                          [37, 16], [37], [37, 16, 14], [38, 28, 30], [38, 27], [38], [38, 28],
                          [39], [37, 18], [40, 15, 6], [40, 15, 7], [40, 15], [40, 16, 12],
                          [40, 16], [40], [40, 16, 14], [41, 30], [41, 31, 27], [41, 31], [41],
                          [41, 32], [40, 17]]

        link_capacities_expected = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 40, 10, 40, 40, 10, 10,
                                    40, 10, 10, 10, 10, 10, 10, 40, 10, 10, 10, 10, 10, 10, 40, 10,
                                    10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

        assert (self.__class__.paths == paths_expected)
        assert (self.__class__.link_capacities == link_capacities_expected)

    def test_d_extract_links(self):
        self.__class__.links, self.__class__.capacities_links = tfr_utils.extract_links(
            self.__class__.num_nodes, self.__class__.connections_lists,
            self.__class__.link_capacity_dict)

        links_expected = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 7), (2, 0), (2, 1), (2, 5),
                          (3, 0), (3, 4), (3, 8), (4, 3), (4, 5), (4, 6), (5, 2), (5, 4), (5, 12),
                          (5, 13), (6, 4), (6, 7), (7, 1), (7, 6), (7, 10), (8, 3), (8, 9), (8, 11),
                          (9, 8), (9, 10), (9, 12), (10, 7), (10, 9), (10, 11), (10, 13), (11, 8),
                          (11, 10), (11, 12), (12, 5), (12, 9), (12, 11), (13, 5), (13, 10)]

        assert (self.__class__.links == links_expected)

    def test_e_get_corresponding_values(self):
        self.__class__.rslt_pos_gnrtr = \
            tfr_utils.ResultsPositionGenerator(self.__class__.num_nodes)
        with open(TEST_CODE_DIR + '/../unit-resources/nsfnetbw/simulationResult.txt') as result_file:
            for rslt_data in result_file:
                self.__class__.result_data = rslt_data.split(',')
                self.__class__.traffic_bw_txs, self.__class__.delays, self.__class__.jitters = \
                    tfr_utils.get_corresponding_values(self.__class__.rslt_pos_gnrtr,
                                                       self.__class__.result_data,
                                                       len(self.__class__.paths))

                traffic_bw_txs_expected = np.array(
                    [0.401414, 0.427143, 0.511114, 0.598457, 0.434657, 0.598614, 0.401743, 0.589629,
                     0.333786, 0.461986, 0.47, 0.315357, 0.346143, 0.251429, 0.6256, 0.102371,
                     0.665243, 0.237643, 0.305086, 0.362086, 0.553214, 0.574657, 0.406286, 0.372529,
                     0.4396, 0.320957, 0.648329, 0.582843, 0.110657, 0.275329, 0.117586, 0.484771,
                     0.0795857, 0.2939, 0.588929, 0.676586, 0.573386, 0.151343, 0.605686, 0.599471,
                     0.665929, 0.367286, 0.568571, 0.573414, 0.3607, 0.402243, 0.544329, 0.488757,
                     0.1364, 0.536086, 0.470414, 0.431971, 0.146571, 0.404214, 0.659457, 0.529571,
                     0.389171, 0.130786, 0.350443, 0.362314, 0.238071, 0.1894, 0.551086, 0.541814,
                     0.358214, 0.213586, 0.444471, 0.147314, 0.0822286, 0.279771, 0.448914,
                     0.162671, 0.444671, 0.193886, 0.459314, 0.305214, 0.6609, 0.624229, 0.499357,
                     0.341143, 0.286429, 0.449529, 0.352129, 0.642486, 0.505757, 0.130629, 0.104386,
                     0.667886, 0.497471, 0.473757, 0.4847, 0.187457, 0.214986, 0.291957, 0.156129,
                     0.529071, 0.277657, 0.446343, 0.310914, 0.268071, 0.428743, 0.0929714,
                     0.336157, 0.459229, 0.670114, 0.662314, 0.134114, 0.479486, 0.2002, 0.472014,
                     0.167343, 0.687057, 0.476643, 0.434457, 0.228529, 0.314429, 0.357843, 0.3666,
                     0.217814, 0.468043, 0.174671, 0.276386, 0.137114, 0.480971, 0.465186, 0.263429,
                     0.152857, 0.554571, 0.188571, 0.657157, 0.296814, 0.480814, 0.583243, 0.0826,
                     0.131329, 0.449471, 0.588257, 0.482714, 0.125943, 0.670114, 0.669743, 0.616343,
                     0.3722, 0.386514, 0.677229, 0.107371, 0.453514, 0.346814, 0.525357, 0.0854143,
                     0.0948571, 0.348, 0.243114, 0.671486, 0.136571, 0.296143, 0.252243, 0.364686,
                     0.133186, 0.497114, 0.271829, 0.614229, 0.331914, 0.638014, 0.115629, 0.213429,
                     0.494557, 0.4486, 0.418986, 0.602386, 0.230686, 0.3758, 0.392514, 0.635029,
                     0.125343, 0.641314, 0.423314, 0.116586, 0.655314, 0.239986, 0.266171, 0.0731])

                delays_expected = np.array(
                    [0.122149, 0.118475, 0.151613, 0.179496, 0.311971, 0.340763, 0.266415, 0.356377,
                     0.495298, 0.292236, 0.497043, 0.472212, 0.481137, 0.112192, 0.117155, 0.267336,
                     0.457612, 0.315428, 0.281226, 0.142632, 0.475086, 0.316982, 0.167622, 0.28693,
                     0.470249, 0.485287, 0.121272, 0.114872, 0.27328, 0.216626, 0.183396, 0.381601,
                     0.251574, 0.489573, 0.480316, 0.488748, 0.460723, 0.34868, 0.362408, 0.153114,
                     0.278829, 0.26956, 0.0272861, 0.055193, 0.187812, 0.322843, 0.205796, 0.332074,
                     0.506439, 0.348777, 0.468727, 0.22032, 0.179938, 0.437629, 0.195694, 0.0272774,
                     0.0277996, 0.154286, 0.304519, 0.230169, 0.321955, 0.321035, 0.365985,
                     0.190605, 0.198082, 0.291989, 0.284275, 0.163518, 0.0573325, 0.0278884,
                     0.190102, 0.331715, 0.262923, 0.274928, 0.298379, 0.271719, 0.160377, 0.16556,
                     0.354045, 0.269797, 0.360809, 0.194641, 0.16706, 0.199916, 0.136097, 0.406102,
                     0.310581, 0.164796, 0.28156, 0.351903, 0.366916, 0.248107, 0.135815, 0.25084,
                     0.344241, 0.309204, 0.343192, 0.1412, 0.302165, 0.171825, 0.028002, 0.142875,
                     0.293206, 0.164326, 0.307403, 0.432646, 0.43286, 0.14863, 0.176716, 0.438208,
                     0.342527, 0.319582, 0.128828, 0.301782, 0.136174, 0.264012, 0.433756, 0.436033,
                     0.33306, 0.457199, 0.276635, 0.301153, 0.290661, 0.325646, 0.190846, 0.122348,
                     0.16491, 0.258241, 0.116208, 0.301358, 0.271975, 0.164732, 0.462612, 0.427008,
                     0.323363, 0.29082, 0.166063, 0.0278544, 0.270823, 0.146552, 0.118553, 0.256566,
                     0.138865, 0.423655, 0.279923, 0.469315, 0.267848, 0.327029, 0.302442, 0.291577,
                     0.145269, 0.109457, 0.256478, 0.116824, 0.121349, 0.258849, 0.459127, 0.447423,
                     0.343872, 0.223193, 0.198275, 0.171694, 0.35501, 0.322202, 0.262763, 0.130631,
                     0.2934, 0.114177, 0.337962, 0.443316, 0.439743, 0.32443, 0.213279, 0.185332,
                     0.153977, 0.345944, 0.15728, 0.397576, 0.273682, 0.130647, 0.254697, 0.316847])

                jitters_expected = np.array(
                    [0.00795296, 0.00758389, 0.0123898, 0.0156348, 0.0395151, 0.0414178, 0.028797,
                     0.0524881, 0.0789477, 0.0332378, 0.0769716, 0.0700673, 0.0721532, 0.0060479,
                     0.00725773, 0.028118, 0.0637338, 0.0412216, 0.0317022, 0.0121093, 0.0730746,
                     0.0367705, 0.0135803, 0.0336058, 0.0721314, 0.075112, 0.00723151, 0.00664425,
                     0.0299505, 0.0258712, 0.020273, 0.0514501, 0.0272745, 0.0782576, 0.0724843,
                     0.0719169, 0.0686392, 0.0492675, 0.0493875, 0.0120821, 0.0286524, 0.0291193,
                     0.00036971, 0.00137318, 0.0175861, 0.0356984, 0.0275158, 0.0461458, 0.0884577,
                     0.0458006, 0.0723692, 0.0214486, 0.0157767, 0.0624766, 0.0184479, 0.00037234,
                     0.00037739, 0.0149136, 0.0360869, 0.0304702, 0.036501, 0.0401286, 0.0504535,
                     0.016781, 0.0200013, 0.031831, 0.030628, 0.0153513, 0.00129068, 0.0003923,
                     0.0174406, 0.0402676, 0.0325893, 0.0305412, 0.0339232, 0.0277576, 0.0141254,
                     0.0154792, 0.0431485, 0.0294415, 0.0442709, 0.0193143, 0.0159383, 0.0184967,
                     0.0100148, 0.0589994, 0.0340969, 0.0131105, 0.0315768, 0.0458041, 0.0486836,
                     0.0254059, 0.0095566, 0.0262766, 0.0389563, 0.037056, 0.0400327, 0.0107409,
                     0.0321863, 0.0138605, 0.00038045, 0.00994583, 0.0316567, 0.012981, 0.0351774,
                     0.0577288, 0.0565234, 0.0134096, 0.0160102, 0.061281, 0.0475795, 0.0381832,
                     0.00898747, 0.0352138, 0.00914876, 0.0268203, 0.0624383, 0.0624367, 0.0381304,
                     0.0674163, 0.0306675, 0.0381671, 0.0348066, 0.0397888, 0.017382, 0.00823402,
                     0.0158138, 0.0272592, 0.00691229, 0.0347312, 0.0293405, 0.0125665, 0.0678425,
                     0.0601201, 0.0400858, 0.0339472, 0.0136206, 0.00038138, 0.0276956, 0.0119842,
                     0.00710855, 0.0268154, 0.0101342, 0.0591072, 0.0318002, 0.0649794, 0.0291805,
                     0.0397603, 0.0354575, 0.0314825, 0.010017, 0.00598866, 0.0268539, 0.00749781,
                     0.00787736, 0.0270116, 0.0642614, 0.0676913, 0.0438315, 0.0230667, 0.0195785,
                     0.0173688, 0.0458108, 0.0391208, 0.0259448, 0.00894155, 0.0339221, 0.00629836,
                     0.0427991, 0.0648765, 0.0638943, 0.04365, 0.0213804, 0.0181983, 0.0149627,
                     0.0447186, 0.0111644, 0.0565135, 0.0288794, 0.00813516, 0.0259787, 0.0374633])

                np.testing.assert_array_equal(self.__class__.traffic_bw_txs,
                                              traffic_bw_txs_expected)
                np.testing.assert_array_equal(self.__class__.delays, delays_expected)
                np.testing.assert_allclose(self.__class__.jitters, jitters_expected, atol=1e-05)

    def test_f_make_indices(self):
        self.__class__.link_indices, self.__class__.path_indices, self.__class__.sequ_indices = \
            tfr_utils.make_indices(self.__class__.paths)

        link_indices_expected = [0, 1, 2, 2, 10, 1, 8, 2, 10, 14, 0, 5, 2, 11, 2, 11, 25, 0, 5, 23,
                                 2, 11, 26, 1, 8, 17, 1, 8, 18, 3, 4, 3, 2, 5, 22, 19, 4, 8, 5, 22,
                                 5, 3, 2, 11, 5, 23, 31, 5, 23, 5, 23, 32, 4, 8, 17, 4, 8, 18, 6, 7,
                                 6, 2, 8, 16, 8, 8, 16, 14, 7, 5, 6, 2, 11, 8, 17, 38, 8, 18, 41, 8,
                                 17, 39, 8, 17, 8, 18, 9, 9, 0, 9, 1, 10, 10, 13, 10, 14, 10, 14,
                                 20, 11, 11, 25, 11, 25, 28, 11, 26, 11, 26, 36, 10, 13, 18, 12, 9,
                                 14, 20, 21, 13, 15, 12, 13, 14, 14, 20, 12, 11, 13, 17, 38, 13, 18,
                                 41, 12, 11, 26, 13, 17, 13, 18, 15, 6, 15, 7, 15, 16, 12, 16, 16,
                                 14, 16, 14, 20, 16, 12, 11, 17, 38, 18, 41, 17, 39, 17, 18, 19, 12,
                                 9, 20, 21, 19, 13, 15, 19, 12, 19, 19, 13, 20, 19, 12, 11, 20, 23,
                                 31, 20, 23, 20, 23, 32, 19, 13, 17, 19, 13, 18, 21, 3, 21, 21, 4,
                                 22, 19, 12, 22, 19, 22, 19, 13, 22, 23, 31, 27, 23, 31, 23, 23, 32,
                                 23, 31, 29, 23, 33, 24, 9, 24, 9, 0, 24, 9, 1, 24, 24, 10, 26, 36,
                                 37, 24, 10, 14, 25, 28, 30, 25, 25, 28, 26, 26, 36, 25, 28, 33, 27,
                                 24, 9, 28, 30, 21, 29, 37, 15, 27, 24, 27, 24, 10, 29, 37, 28, 30,
                                 22, 28, 30, 27, 28, 27, 26, 29, 28, 33, 30, 21, 3, 30, 21, 33, 40,
                                 15, 31, 27, 24, 33, 40, 16, 33, 40, 30, 22, 30, 31, 27, 31, 32, 31,
                                 29, 33, 34, 24, 9, 35, 30, 21, 36, 37, 15, 34, 24, 36, 37, 16, 36,
                                 37, 35, 30, 22, 35, 30, 34, 36, 38, 35, 36, 35, 33, 37, 15, 6, 37,
                                 15, 7, 37, 15, 37, 16, 12, 37, 16, 37, 37, 16, 14, 38, 28, 30, 38,
                                 27, 38, 38, 28, 39, 37, 18, 40, 15, 6, 40, 15, 7, 40, 15, 40, 16,
                                 12, 40, 16, 40, 40, 16, 14, 41, 30, 41, 31, 27, 41, 31, 41, 41, 32,
                                 40, 17]
        path_indices_expected = [0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10,
                                 10, 11, 11, 11, 12, 12, 12, 13, 14, 15, 15, 16, 16, 16, 17, 17, 18,
                                 18, 19, 20, 20, 20, 21, 21, 21, 22, 22, 23, 23, 23, 24, 24, 24, 25,
                                 25, 25, 26, 27, 28, 28, 29, 29, 30, 31, 31, 31, 32, 32, 33, 33, 33,
                                 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 38, 38, 39, 40, 40, 41,
                                 41, 42, 43, 43, 44, 44, 45, 45, 45, 46, 47, 47, 48, 48, 48, 49, 49,
                                 50, 50, 50, 51, 51, 51, 52, 52, 53, 53, 53, 54, 54, 55, 56, 57, 58,
                                 58, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 64, 64, 65,
                                 65, 66, 66, 67, 68, 68, 69, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73,
                                 74, 74, 75, 75, 76, 77, 78, 78, 78, 79, 79, 80, 80, 80, 81, 81, 82,
                                 83, 83, 84, 85, 85, 85, 86, 86, 86, 87, 87, 88, 88, 88, 89, 89, 89,
                                 90, 90, 90, 91, 91, 92, 93, 93, 94, 94, 94, 95, 95, 96, 96, 96, 97,
                                 98, 98, 98, 99, 99, 100, 101, 101, 102, 102, 102, 103, 103, 104,
                                 104, 105, 105, 105, 106, 106, 106, 107, 108, 108, 109, 109, 109,
                                 110, 110, 110, 111, 111, 111, 112, 113, 113, 114, 115, 115, 116,
                                 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120,
                                 121, 121, 121, 122, 122, 123, 123, 123, 124, 124, 125, 126, 127,
                                 127, 128, 129, 129, 130, 130, 130, 131, 131, 132, 132, 132, 133,
                                 133, 133, 134, 134, 134, 135, 135, 136, 136, 137, 138, 138, 139,
                                 140, 141, 141, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145,
                                 146, 146, 147, 147, 147, 148, 148, 149, 149, 149, 150, 150, 151,
                                 152, 152, 153, 154, 155, 155, 156, 156, 156, 157, 157, 157, 158,
                                 158, 159, 159, 159, 160, 160, 161, 162, 162, 162, 163, 163, 163,
                                 164, 164, 165, 166, 166, 167, 168, 168, 169, 169, 169, 170, 170,
                                 170, 171, 171, 172, 172, 172, 173, 173, 174, 175, 175, 175, 176,
                                 176, 177, 177, 177, 178, 178, 179, 180, 180, 181, 181]
        sequ_indices_expected = [0, 0, 0, 0, 1, 0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1,
                                 2, 0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1, 2,
                                 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 0, 1, 0, 0,
                                 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 0,
                                 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2,
                                 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 2, 0, 1,
                                 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 2, 0,
                                 1, 2, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 0,
                                 1, 0, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0,
                                 0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 0, 1, 2, 0, 1, 0, 0, 1, 0, 1, 2,
                                 0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                                 0, 0, 1, 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1,
                                 2, 0, 1, 0, 1, 2, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 2, 0, 1, 0, 1,
                                 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 2,
                                 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1, 0,
                                 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1, 2, 0, 1,
                                 2, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1,
                                 0, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1, 0, 1]

        assert (self.__class__.link_indices == link_indices_expected)
        assert (self.__class__.path_indices == path_indices_expected)
        assert (self.__class__.sequ_indices == sequ_indices_expected)

    def test_g_make_tfrecord(self):

        tfr_utils.make_tfrecord(TEST_CODE_DIR + '/../unit-resources/nsfnetbw/',
                                'test_results.tfrecords',
                                self.__class__.connections_lists,
                                self.__class__.num_nodes,
                                self.__class__.link_capacity_dict,
                                self.__class__.routing_mtrx,
                                open(TEST_CODE_DIR +
                                     '/../unit-resources/nsfnetbw/simulationResult.txt',
                                     mode='rb'))

        assert(os.path.exists(self.tf_rcrds_fl_nm))
        os.remove(self.tf_rcrds_fl_nm)

    def test_h_write_tfrecord(self):
        num_paths = len(self.__class__.paths)
        num_links = max(max(self.__class__.paths)) + 1
        n_total = len(self.__class__.path_indices)
        tf_rcrd_wrtr = tf.io.TFRecordWriter(self.tf_rcrds_fl_nm)

        tfr_utils.write_tfrecord(self.__class__.result_data,
                                 self.__class__.rslt_pos_gnrtr,
                                 num_paths,
                                 self.__class__.link_capacities,
                                 self.__class__.link_indices,
                                 self.__class__.path_indices,
                                 self.__class__.sequ_indices,
                                 num_links,
                                 n_total,
                                 tf_rcrd_wrtr)

        assert(os.path.exists(self.tf_rcrds_fl_nm))

    """
    This code is left here for pedagogical purposes. See:
    https://stackoverflow.com/questions/57725172/iterating-over-a-dataset-tf-2-0-with-for-loop
    TODO revisit after upgrading code for TF 2.0 eager execution idioms.
    """
    def test_i_read_dataset(self):
        with tf.compat.v1.Session() as sess:
            data_set = rn_train.read_dataset(self.tf_rcrds_fl_nm)
            itrtr = tf.compat.v1.data.make_initializable_iterator(data_set)
            sess.run(itrtr.initializer)
            features, labels = itrtr.get_next()
            features_keys = features.keys()

            assert(len(features_keys) == 8)

            test_dict_keys = {'traffic',
                              'sequences',
                              'link_capacity',
                              'links',
                              'paths',
                              'n_links',
                              'n_paths',
                              'n_total'}

            assert (set(features_keys) == test_dict_keys)

            labels_val = labels.eval()
            labels_val = (labels_val[0] * 0.54) + 0.37
            np.testing.assert_allclose(labels_val, np.array(self.__class__.delays), atol=1e-05)

            sess.run(itrtr.initializer)
            links_val = features['links'].eval()
            assert(links_val.tolist()[0] == self.__class__.link_indices)

            sess.run(itrtr.initializer)
            paths_val = features['paths'].eval()
            assert(paths_val.tolist()[0] == self.__class__.path_indices)

            sess.run(itrtr.initializer)
            sequences_val = features['sequences'].eval()
            assert(sequences_val.tolist()[0] == self.__class__.sequ_indices)

            sess.run(itrtr.initializer)
            n_links_val = features['n_links'].eval()
            assert(n_links_val.tolist()[0] == max(max(self.__class__.paths)) + 1)

            sess.run(itrtr.initializer)
            n_paths_val = features['n_paths'].eval()
            assert(n_paths_val.tolist()[0] == len(self.__class__.paths))

            sess.run(itrtr.initializer)
            n_total_val = features['n_total'].eval()
            assert (n_total_val.tolist()[0] == len(self.__class__.path_indices))

            sess.run(itrtr.initializer)
            traffic_val = features['traffic'].eval()
            traffic_val = (traffic_val[0] * 0.13) + 0.17
            np.testing.assert_allclose(traffic_val, np.array(self.__class__.traffic_bw_txs),
                                       atol=1e-05)

            sess.run(itrtr.initializer)
            link_capacity_val = features['link_capacity'].eval()
            link_capacity_val = (link_capacity_val[0] * 40.0) + 25.0
            np.testing.assert_array_equal(link_capacity_val,
                                          np.array(self.__class__.link_capacities))
        os.remove(self.tf_rcrds_fl_nm)

    """
    def test_ia_read_dataset(self):
        tf.compat.v1.enable_eager_execution()
        data_set = rn_train.read_dataset(self.tf_rcrds_fl_nm)
        for features, labels in data_set:
            features_keys = features.keys()
            labels_array = np.array(labels)

            assert(len(features_keys) == 8)

            test_dict_keys = {'traffic',
                              'sequences',
                              'link_capacity',
                              'links',
                              'paths',
                              'n_links',
                              'n_paths',
                              'n_total'}

            assert (set(features_keys) == test_dict_keys)

            labels_array = (labels_array[0] * 0.54) + 0.37
            np.testing.assert_allclose(labels_array, np.array(self.__class__.delays), atol=1e-05)

            links_val = np.array(features['links'])
            np.testing.assert_array_equal(links_val[0], self.__class__.link_indices)

            paths_val = np.array(features['paths'])
            np.testing.assert_array_equal(paths_val[0], self.__class__.path_indices)

            sequences_val = np.array(features['sequences'])
            np.testing.assert_array_equal(sequences_val[0], self.__class__.sequ_indices)

            n_links_val = np.array(features['n_links'])
            np.testing.assert_array_equal(n_links_val[0], max(max(self.__class__.paths)) + 1)

            n_paths_val = np.array(features['n_paths'])
            np.testing.assert_array_equal(n_paths_val[0], len(self.__class__.paths))

            n_total_val = np.array(features['n_total'])
            np.testing.assert_array_equal(n_total_val[0], len(self.__class__.path_indices))

            traffic_val = np.array(features['traffic'])
            traffic_val = (traffic_val[0] * 0.13) + 0.17
            np.testing.assert_allclose(traffic_val, np.array(self.__class__.traffic_bw_txs),
                                       atol=1e-05)

            link_capacity_val = np.array(features['link_capacity'])
            link_capacity_val = (link_capacity_val[0] * 40.0) + 25.0
            np.testing.assert_array_equal(link_capacity_val,
                                          np.array(self.__class__.link_capacities))
        os.remove(self.tf_rcrds_fl_nm)
    """

    def test_j_process_data(self):
        tfr_utils.process_data(self.network_data_dir, te_split=0.5)
        # Which file ends up in which directory varies between macOS and Linux.
        assert(os.path.exists(self.tf_rcrds_dir +
                              '/train/results_nsfnetbw_9_Routing_SP_k_0.tfrecords') |
               os.path.exists(self.tf_rcrds_dir +
                              '/train/results_nsfnetbw_9_Routing_SP_k_1.tfrecords'))
        assert (os.path.exists(self.tf_rcrds_dir +
                               '/evaluate/results_nsfnetbw_9_Routing_SP_k_0.tfrecords') |
                os.path.exists(self.tf_rcrds_dir +
                               '/evaluate/results_nsfnetbw_9_Routing_SP_k_1.tfrecords'))
        shutil.rmtree(self.tf_rcrds_dir, ignore_errors=True)


    @classmethod
    def tearDownClass(cls):
        pass
