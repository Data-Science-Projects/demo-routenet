#!/usr/bin/env python3
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

"""
This script calculates normalisation values for datasets from datasets_v0.
"""

from routenet.train.normvals_v0 import NormVals

datasets = ['nsfnetbw', 'geant2bw', 'synth50bw']

norm_vals = NormVals()

norm_vals.calculate_norm_vals(datasets)
norm_vals.save_norm_params()
