#!/usr/bin/env bash

unset PYTHONPATH
python3 -m venv routenet
. routenet/bin/activate
export PYTHONPATH=$PWD/../src:$PWD/../tests
pip install --upgrade pip
pip install -r ../requirements.txt
pip install pytest
