# -*- coding: utf-8 -*-
# Copyright (c) 2019, Krzysztof Rusek [^1], Paul Almasan [^2]
#
# [^1]: AGH University of Science and Technology, Department of
#     communications, Krakow, Poland. Email: krusek\@agh.edu.pl
#
# [^2]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: almasan@ac.upc.edu


class NewParser:
    _net_size = 0
    _offset_delay = 0
    has_packet_gen = True

    def __init__(self, net_size):
        self._net_size = net_size
        self._offset_delay = net_size * net_size * 3

    def get_bw_ptr(self, src, dst):
        return (src * self._net_size + dst) * 3

    def get_gen_pckt_ptr(self, src, dst):
        return (src * self._net_size + dst) * 3 + 1

    def get_drop_pckt_ptr(self, src, dst):
        return (src * self._net_size + dst) * 3 + 2

    def get_delay_ptr(self, src, dst):
        return self._offset_delay + (src * self._net_size + dst) * 7

    def get_jitter_ptr(self, src, dst):
        return self._offset_delay + (src * self._net_size + dst) * 7 + 6
