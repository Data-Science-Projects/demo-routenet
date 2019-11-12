#!/usr/bin/env python3
__copyright__ = 'Copyright 2019 Nathan Sowatskey, Ana Matute. All rights reserved.'
__author__ = 'nsowatsk@cisco.com'

"""
This script trains, and creates checkpoints for, the RouteNetModel with the datasets from datasets_v0.
"""

import datetime
import glob
import os
import shutil

from routenet.model.routenet_model_v0 import RouteNetModelV0
from routenet.train import train_v0 as rn_train

date_str = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())

omnet_data_dir = os.getenv('OMNET_DATA_DIR')

train_sets = ['nsfnetbw', 'geant2bw']
eval_sets = ['synth50bw']

train_files_list = []
eval_files_list = []
model_chkpnt_dir = '../model_checkpoints-train'

warm_start = False
warm_start_dir = None

for name in train_sets:

    files_list = glob.glob(omnet_data_dir +
                                 '/datasets_v0/' + name + '/tfrecords/train/*.tfrecords')
    train_files_list = train_files_list + files_list

    model_chkpnt_dir = model_chkpnt_dir + '_' + name

model_chkpnt_dir = model_chkpnt_dir + '-eval'

for name in eval_sets:

    files_list = glob.glob(omnet_data_dir +
                                         '/datasets_v0/' + name + '/tfrecords/evaluate/*.tfrecords')
    eval_files_list = eval_files_list + files_list

    model_chkpnt_dir = model_chkpnt_dir + '_' + name

train_steps = 100000

model_chkpnt_dir = model_chkpnt_dir + '-' + str(train_steps) + '_v0_' + date_str + '/'

if not warm_start:
    shutil.rmtree(model_chkpnt_dir, ignore_errors=True)
else:
    warm_start_dir = model_chkpnt_dir

rn_train.train_and_evaluate(model_dir=model_chkpnt_dir,
                            train_files=train_files_list,
                            shuffle_buf=30000,
                            target='delay',
                            train_steps=train_steps,
                            eval_files=eval_files_list,
                            warm_start_from=warm_start_dir,
                            model_hparams=RouteNetModelV0.default_hparams,
                            checkpointing_config=rn_train.rn_default_checkpointing_config)
