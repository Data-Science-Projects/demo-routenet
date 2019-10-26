#!/usr/bin/env bash
# First get the data files and populate the base data directory.
# These data sets are from:
# https://github.com/knowledgedefinednetworking/NetworkModelingDatasets/tree/master/datasets_v0
rm -rf $OMNET_DATA_DIR
mkdir -p $OMNET_DATA_DIR/datasets_v0
cd $OMNET_DATA_DIR/datasets_v0
wget "http://knowledgedefinednetworking.org/data/datasets_v0/nsfnet.tar.gz"
wget "http://knowledgedefinednetworking.org/data/datasets_v0/geant2.tar.gz"
wget "http://knowledgedefinednetworking.org/data/datasets_v0/synth50.tar.gz"
tar -xvzf nsfnet.tar.gz
tar -xvzf geant2.tar.gz
tar -xvzf synth50.tar.gz

# These data sets are from:
# https://github.com/knowledgedefinednetworking/Unveiling-the-potential-of-GNN-for-network-modeling-and-optimization-in-SDN/tree/master/datasets
mkdir -p $OMNET_DATA_DIR/datasets
cd $OMNET_DATA_DIR/datasets
wget "http://knowledgedefinednetworking.org/data/nsfnet.zip"
wget "http://knowledgedefinednetworking.org/data/geant2.zip"
wget "http://knowledgedefinednetworking.org/data/GBN.zip"
unzip nsfnet.zip
unzip geant2.zip
unzip GBN.zip

# Populate the smoke-resources data sets for smoke-testing
rm -rf $RN_PROJ_HOME/tests/smoke-resources/data

mkdir -p $RN_PROJ_HOME/tests/smoke-resources/data/nsfnetbw
cp $OMNET_DATA_DIR/datasets_v0/nsfnetbw/graph_attr.txt $RN_PROJ_HOME/tests/smoke-resources/data/nsfnetbw
cp $OMNET_DATA_DIR/datasets_v0/nsfnetbw/Network_nsfnetbw.ned $RN_PROJ_HOME/tests/smoke-resources/data/nsfnetbw
cp $OMNET_DATA_DIR/datasets_v0/nsfnetbw/results_nsfnetbw_9_Routing_SP* $RN_PROJ_HOME/tests/smoke-resources/data/nsfnetbw

mkdir -p $RN_PROJ_HOME/tests/smoke-resources/data/geant2bw
cp $OMNET_DATA_DIR/datasets_v0/geant2bw/graph_attr.txt $RN_PROJ_HOME/tests/smoke-resources/data/geant2bw
cp $OMNET_DATA_DIR/datasets_v0/geant2bw/Network_geant2bw.ned $RN_PROJ_HOME/tests/smoke-resources/data/geant2bw
cp $OMNET_DATA_DIR/datasets_v0/geant2bw/results_geant2bw_9_Routing_SP* $RN_PROJ_HOME/tests/smoke-resources/data/geant2bw

mkdir -p $RN_PROJ_HOME/tests/smoke-resources/data/synth50bw
cp $OMNET_DATA_DIR/datasets_v0/synth50bw/graph_attr.txt $RN_PROJ_HOME/tests/smoke-resources/data/synth50bw
cp $OMNET_DATA_DIR/datasets_v0/synth50bw/Network_synth50bw.ned $RN_PROJ_HOME/tests/smoke-resources/data/synth50bw
cp $OMNET_DATA_DIR/datasets_v0/synth50bw/results_synth50bw_9_Routing_SP* $RN_PROJ_HOME/tests/smoke-resources/data/synth50bw