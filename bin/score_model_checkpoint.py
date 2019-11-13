#!/usr/bin/env python3
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

"""
This script scores checkpoints for the RouteNetModel with the datasets from datasets_v0.
"""

import os

import rn_test_utils.test_utils as test_utils

# Get a sample
sample_file = test_utils.get_sample('synth50bw')

# Then create a RoutNetModel instance and initialise its readout layer with a sample from the
# `sample_file`.
pred_graph, pred_readout, itrtr, labels = test_utils.get_model_readout(sample_file)

# We now have a readout Sequential model initialised with the sample data, and so the network that
# data represents, with training set to True, within the `pred_graph` Graph.

# We can now create a session with that graph, and restore into the session the variables from a
# model checkpoint, i.e. we can transfer the weights from the previously trained model.
proj_dir = os.getenv('RN_PROJ_DIR')

models_dir = proj_dir + '/trained_models/model_checkpoints-train_nsfnetbw_geant2bw-eval_synth50bw-50000_v0_2019-11-12_16-22-33'

median_preds, predicted_delays, true_delay, mse, r2 = test_utils.run_predictions(pred_graph,
                                                                                 pred_readout,
                                                                                 itrtr,
                                                                                 labels,
                                                                                 50000,
                                                                                 models_dir)
print('models_dir - ' + models_dir)
print('mse - ' + str(mse))
print('r2 - ' + str(r2))
