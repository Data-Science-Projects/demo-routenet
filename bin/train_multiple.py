#!/usr/bin/env python3

"""
This script will train, and create checkpoints for, the RouteNetModel with the nsfnetbw and
synth50bw data sets from datasets_v0.
"""

import glob
import os
import shutil

from routenet.train import train as rn_train

NORM_DELAY = True

omnet_data_dir = os.getenv('OMNET_DATA_DIR')

train_sets = ['nsfnetbw', 'geant2bw']

train_files_list = []
eval_files_list = []
model_chkpnt_dir = '../model_checkpoints'

for name in train_sets:

    files_list = glob.glob(omnet_data_dir +
                                 '/datasets_v0/' + name + '/tfrecords/train/*.tfrecords')
    train_files_list = train_files_list + files_list

    files_list = glob.glob(omnet_data_dir +
                                         '/datasets_v0/' + name + '/tfrecords/evaluate/*.tfrecords')
    eval_files_list = eval_files_list + files_list

    model_chkpnt_dir = model_chkpnt_dir + '_' + name

train_steps = 50000

model_chkpnt_dir = model_chkpnt_dir + '_v0/'

shutil.rmtree(model_chkpnt_dir, ignore_errors=True)

rn_train.train_and_evaluate(model_dir=model_chkpnt_dir,
                            train_files=train_files_list,
                            shuffle_buf=30000,
                            target='delay',
                            train_steps=train_steps,
                            eval_files=eval_files_list,
                            warm_start_from=None)
