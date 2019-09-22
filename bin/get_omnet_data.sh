#!/usr/bin/env bash
cd $OMNET_DATA_DIR
wget "http://knowledgedefinednetworking.org/data/datasets_v0/nsfnet.tar.gz"
wget "http://knowledgedefinednetworking.org/data/datasets_v0/geant2.tar.gz"
wget "http://knowledgedefinednetworking.org/data/datasets_v0/synth50.tar.gz"
wget "http://knowledgedefinednetworking.org/data/GBN.zip"
tar -xvzf nsfnet.tar.gz
tar -xvzf geant2.tar.gz
tar -xvzf synth50.tar.gz
unzip GBN.zip