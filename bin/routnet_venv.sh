#!/usr/bin/env bash

python3 -m venv routenet
. routenet/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/../src
pip install --upgrade pip
pip install -r ../requirements.txt
