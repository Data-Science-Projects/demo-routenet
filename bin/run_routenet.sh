#!/bin/bash
PROJECT_DIR=../..
DATA_DIR=$PROJECT_DIR/data/
export PYTHONPATH=$PROJECT_DIR/demo-routenet/src

if [[ "$1" = "tfrecords" ]]; then

    python3 $PYTHONPATH/routenet/routenet_with_link_cap.py data -d $DATA_DIR/$2/

fi

if [[ "$1" = "train" ]]; then

    python3 $PYTHONPATH/routenet/routenet_with_link_cap.py train \
    --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8" \
     --train  $DATA_DIR/$2/tfrecords/train/*.tfrecords \
     --train_steps $3 \
     --eval_ $DATA_DIR/$2/tfrecords/evaluate/*.tfrecords --model_dir ./CheckPoints/$2

fi

if [[ "$1" = "train_multiple" ]]; then

    python3 $PYTHONPATH/routenet/routenet_with_link_cap.py train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8"  --train  $DATA_DIR/$2/tfrecords/train/*.tfrecords $DATA_DIR/$3/tfrecords/train/*.tfrecords --train_steps $5 --eval_ $DATA_DIR/geant2bw/tfrecords/evaluate/*.tfrecords $DATA_DIR/geant2bw/tfrecords/train/*.tfrecords --shuffle_buf 30000 --model_dir ./CheckPoints/$4
fi

if [[ "$1" = "normalize" ]]; then

    python3 $PYTHONPATH/routenet/normalize.py --dir $DATA_DIR/nsfnetbw/tfrecords/train/ $DATA_DIR/nsfnetbw/tfrecords/evaluate/ $DATA_DIR/synth50bw2/tfrecords/evaluate/ $DATA_DIR/synth50bw2/tfrecords/train/ --ini configNSFNET50.ini
fi
