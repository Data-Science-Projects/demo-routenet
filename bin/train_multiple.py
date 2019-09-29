#!/usr/bin/env python3

"""
This script will train, and create checkpoints for, the RouteNetModel with the nsfnetbw and
synth50bw data sets.
"""

import glob
import os
import shutil

from routenet.train import train as rn_train

omnet_data_dir = os.getenv('OMNET_DATA_DIR')

nsfnetbw_train_files_list = glob.glob(omnet_data_dir + '/nsfnetbw/tfrecords/train/*.tfrecords')
synth50bw_train_files_list = glob.glob(omnet_data_dir + '/synth50bw/tfrecords/train/*.tfrecords')
train_files_list = nsfnetbw_train_files_list + synth50bw_train_files_list

nsfnetbw_eval_files_list = glob.glob(omnet_data_dir + '/nsfnetbw/tfrecords/evaluate/*.tfrecords')
synth50bw_eval_files_list = glob.glob(omnet_data_dir + '/synth50bw/tfrecords/evaluate/*.tfrecords')
eval_files_list = nsfnetbw_eval_files_list + synth50bw_eval_files_list

model_chkpnt_dir = '../model_checkpoints-imac/'

shutil.rmtree(model_chkpnt_dir, ignore_errors=True)

rn_train.train_and_evaluate(model_dir=model_chkpnt_dir,
                            train_files=train_files_list,
                            shuffle_buf=30000,
                            target='delay',
                            train_steps=100000,
                            eval_files=eval_files_list,
                            warm_start_from=None)
