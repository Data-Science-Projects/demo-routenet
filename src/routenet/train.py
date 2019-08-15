# -*- coding: utf-8 -*-
# Copyright (c) 2019, Krzysztof Rusek [^1], Paul Almasan [^2]
#
# [^1]: AGH University of Science and Technology, Department of
#     communications, Krakow, Poland. Email: krusek\@agh.edu.pl
#
# [^2]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: almasan@ac.upc.edu


from __future__ import print_function

import argparse

import tensorflow as tf

from routenet.comnet_model import ComnetModel
from routenet.dataset_utils import parse
from routenet.tfrecords import data


def cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes)-1):
            cummaxes.append(tf.math.add_n(maxes[0:i+1]))

    return cummaxes


def transformation_func(itrtr, batch_size=32):
    with tf.name_scope('transformation_func'):
        vs = [itrtr.get_next() for _ in range(batch_size)]

        links_cummax = cummax(vs, lambda v: v[0]['links'])
        paths_cummax = cummax(vs, lambda v: v[0]['paths'])

        tensors = ({'traffic': tf.concat([v[0]['traffic'] for v in vs], axis=0),
                    'sequences': tf.concat([v[0]['sequences'] for v in vs], axis=0),
                    'link_capacity': tf.concat([v[0]['link_capacity'] for v in vs], axis=0),
                    'links': tf.concat([v[0]['links'] + m for v, m in zip(vs, links_cummax)],
                                       axis=0),
                    'paths': tf.concat([v[0]['paths'] + m for v, m in zip(vs, paths_cummax)],
                                       axis=0),
                    'n_links': tf.math.add_n([v[0]['n_links'] for v in vs]),
                    'n_paths': tf.math.add_n([v[0]['n_paths'] for v in vs]),
                    'n_total': tf.math.add_n([v[0]['n_total'] for v in vs])
                    }, tf.concat([v[1] for v in vs], axis=0))

    return tensors


def tfrecord_input_fn(filenames, hparams, shuffle_buf=1000, target='delay'):
    
    files = tf.data.Dataset.from_tensor_slices(filenames)
    files = files.shuffle(len(filenames))

    ds = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4))

    if shuffle_buf:
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(shuffle_buf))

    ds = ds.map(lambda buf: parse(buf, target), num_parallel_calls=2)
    ds = ds.prefetch(10)

    itrtr = ds.make_one_shot_iterator()
    sample = transformation_func(itrtr, hparams.batch_size)

    return sample


def model_fn(features, labels, mode, params):
    """
    TBD
    :param features: This is batch_features from input_fn
    :param labels: This is batch_labrange
    :param mode: An instance of tf.estimator.ModeKeys
    :param params: Additional configuration
    :return: TBD
    """

    model = ComnetModel(params)
    model.build()

    def fn(x):
        r = model(x, training=mode == tf.estimator.ModeKeys.TRAIN)
        return r

    predictions = fn(features)

    predictions = tf.squeeze(predictions)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'predictions': predictions})

    loss = tf.losses.mean_squared_error(labels=labels,
                                        predictions=predictions,
                                        reduction=tf.losses.Reduction.MEAN)

    regularization_loss = sum(model.losses)
    total_loss = loss + regularization_loss

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('regularization_loss', regularization_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss,
            eval_metric_ops={
                'label/mean': tf.metrics.mean(labels),
                'prediction/mean': tf.metrics.mean(predictions),
                'mae': tf.metrics.mean_absolute_error(labels, predictions),
                'rho': tf.contrib.metrics.streaming_pearson_correlation(labels=labels,
                                                                        predictions=predictions),
                'mre': tf.metrics.mean_relative_error(labels, predictions, labels)
            }
        )

    assert mode == tf.estimator.ModeKeys.TRAIN

    trainables = model.variables
    grads = tf.gradients(total_loss, trainables)
    grad_var_pairs = zip(grads, trainables)

    summaries = [tf.summary.histogram(var.op.name, var) for var in trainables]
    summaries += [tf.summary.histogram(g.op.name, g) for g in grads if g is not None]

    decayed_lr = tf.train.exponential_decay(params.learning_rate,
                                            tf.train.get_global_step(), 82000,
                                            0.8, staircase=True)
    optimizer = tf.train.AdamOptimizer(decayed_lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grad_var_pairs, global_step=tf.train.get_global_step())

    logging_hook = tf.train.LoggingTensorHook(
        {"Training loss": loss}, every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=[logging_hook]
                                      )



# TODO this is a global hparams, which should be factored into a non-global variable or class. These
# are not the same as the values passed in for training, or those used in the demo notebook. This
# needs to be redone.

default_hparams = tf.contrib.training.HParams(link_state_dim=32,
                                              path_state_dim=32,
                                              T=8,
                                              readout_units=256,
                                              learning_rate=0.001,
                                              batch_size=32,
                                              dropout_rate=0.5,
                                              l2=0.1,
                                              l2_2=0.01,
                                              learn_embedding=True)  # If false, only the readout is trained


def train(parsed_args):
    print(parsed_args)
    tf.logging.set_verbosity('INFO')

    if parsed_args.hparams:
        default_hparams.parse(parsed_args.hparams)

    print('args:', args)    

    process_train(model_hparams=default_hparams,
                  model_dir=args.model_dir,
                  train_files=args.train,
                  shuffle_buf=args.shuffle_buf,
                  target=args.target,
                  train_steps=args.train_steps,
                  eval_files=args.eval_,
                  warm_start_from=args.warm)


def process_train(model_hparams,
                  model_dir,
                  train_files,
                  shuffle_buf,
                  target,
                  train_steps,
                  eval_files,
                  warm_start_from):

    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=10*60,  # Save checkpoints every 10 minutes
        keep_checkpoint_max=20  # Retain the 10 most recent checkpoints.
    )
    

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=model_dir,
                                       params=model_hparams,
                                       warm_start_from=warm_start_from,
                                       config=my_checkpointing_config)

    train_spec = tf.estimator.TrainSpec(input_fn=
                                        lambda: tfrecord_input_fn(filenames=train_files,
                                                                  hparams=model_hparams,
                                                                  shuffle_buf=shuffle_buf,
                                                                  target=target),
                                        max_steps=train_steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=
                                      lambda: tfrecord_input_fn(filenames=eval_files,
                                                                hparams=model_hparams,
                                                                shuffle_buf=None,
                                                                target=target),
                                      throttle_secs=10*60)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RouteNet: a Graph Neural Network '
                                                 'model for computer network modeling')

    subparsers = parser.add_subparsers(help='sub-command help')
    parser_data = subparsers.add_parser('data', help='data processing')
    parser_data.add_argument('-d', help='data file', type=str, required=True, nargs='+')
    parser_data.set_defaults(func=data)

    parser_train = subparsers.add_parser('train', help='Train options')
    parser_train.add_argument('--hparams', type=str, help='Comma separated list of '
                                                          '"name=value" pairs.')
    parser_train.add_argument('--train', help='Train Tfrecords files', type=str, nargs='+')
    parser_train.add_argument('--eval_', help='Evaluation Tfrecords files', type=str, nargs='+')
    parser_train.add_argument('--model_dir', help='Model directory', type=str)
    parser_train.add_argument('--train_steps', help='Training steps', type=int, default=100)
    parser_train.add_argument('--eval_steps', help='Evaluation steps, defaul None= all',
                              type=int, default=None)
    parser_train.add_argument('--shuffle_buf', help="Buffer size for samples shuffling",
                              type=int, default=10000)
    parser_train.add_argument('--target', help="Predicted variable", type=str, default='delay')
    parser_train.add_argument('--warm', help="Warm start from", type=str, default=None)
    parser_train.set_defaults(func=train)
    parser_train.set_defaults(name="Train")

    args = parser.parse_args()
    args.func(args)
