#!/usr/bin/env bash

cd $RN_PROJ_HOME/tests/unit
pytest -s test_omnet_tfrecord_utils.py
pytest -s test_train.py
pytest -s test_model.py

cd $RN_PROJ_HOME/tests/smoke_tests
pytest -s omnet_smoke_tests.py


