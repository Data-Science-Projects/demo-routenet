#!/usr/bin/env bash

unset PYTHONPATH
rm -rf routenet_venv
python3 -m venv routenet_venv
. ./set_routenet_venv.sh
pip install --upgrade pip
pip install -r ../requirements.txt
