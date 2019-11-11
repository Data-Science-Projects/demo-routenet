#!/usr/bin/env bash

cd $RN_PROJ_HOME/tests/unit
pytest -s test_omnet_tfrecord_utils_v0.py
pytest -s test_train_v0.py
pytest -s test_model_v0.py

cd $RN_PROJ_HOME/tests/smoke_tests
pytest -s omnet_smoke_tests_v0.py


