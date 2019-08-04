# -*- coding: utf-8 -*-
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

import unittest

from routenet import routenet_with_link_cap as rn


class SmokeTest(unittest.TestCase):

    def test_nsfnetbw(self):
        rn.process_data(data_dir_path='../../../data/nsfnetbw')


