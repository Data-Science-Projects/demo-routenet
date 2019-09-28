#!/usr/bin/env python3

"""
This script will train, and create checkpoints for, the RouteNetModel with the nsfnetbw and
synth50bw data sets.
"""

import glob
import os

from routenet.train import train as rn_train

omnet_data_dir = os.getenv('OMNET_DATA_DIR')

nsfnetbw_train_files_list = glob.glob(omnet_data_dir + '/nsfnetbw/tfrecords/train/*.tfrecords')
synth50bw_train_files_list = glob.glob(omnet_data_dir + '/synth50bw/tfrecords/train/*.tfrecords')

nsfnetbw_eval_files_list = glob.glob(omnet_data_dir + '/nsfnetbw/tfrecords/evaluate/*.tfrecords')
synth50bw_eval_files_list = glob.glob(omnet_data_dir + '/synth50bw/tfrecords/evaluate/*.tfrecords')

rn_train.train_and_evaluate(model_dir='../trained_models',
                            train_files=nsfnetbw_train_files_list + synth50bw_train_files_list,
                            shuffle_buf=30,
                            target='delay',
                            train_steps=100,
                            eval_files=nsfnetbw_eval_files_list + synth50bw_eval_files_list,
                            warm_start_from=None)
