# -*- coding: utf-8 -*-
# Copyright (c) 2019, Krzysztof Rusek [^1], Paul Almasan [^2]
#
# [^1]: AGH University of Science and Technology, Department of
#     communications, Krakow, Poland. Email: krusek\@agh.edu.pl
#
# [^2]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: almasan@ac.upc.edu

"""
The functions here execute the training steps for the RouteNetModel.
"""

from __future__ import print_function

import sys

import tensorflow as tf

from routenet.model.routenet_model import RouteNetModel

rn_default_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs=10 * 60,  # Save checkpoints every 10 minutes
    keep_checkpoint_max=20  # Retain the 20 most recent checkpoints.
)


def train_and_evaluate(model_dir,
                       train_files,
                       shuffle_buf,
                       target,
                       train_steps,
                       eval_files,
                       warm_start_from,
                       model_hparams=RouteNetModel.default_hparams,
                       checkpointing_config=rn_default_checkpointing_config):

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=model_dir,
                                       params=model_hparams,
                                       warm_start_from=warm_start_from,
                                       config=checkpointing_config)

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
                                      throttle_secs=10 * 60)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def cummax(batch_vals, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(val)) + 1 for val in batch_vals]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes)-1):
            cummaxes.append(tf.math.add_n(maxes[0:i+1]))

    return cummaxes


def transformation_func(itrtr, batch_size=32):
    with tf.name_scope('transformation_func'):
        batch_vals = [itrtr.get_next() for _ in range(batch_size)]

        links_cummax = cummax(batch_vals, lambda val: val[0]['links'])
        paths_cummax = cummax(batch_vals, lambda val: val[0]['paths'])

        tensors = ({'traffic': tf.concat([val[0]['traffic'] for val in batch_vals], axis=0),
                    'sequences': tf.concat([val[0]['sequences'] for val in batch_vals], axis=0),
                    'link_capacity': tf.concat([val[0]['link_capacity'] for val in batch_vals],
                                               axis=0),
                    'links': tf.concat([val[0]['links'] + m for val, m in zip(batch_vals,
                                                                              links_cummax)],
                                       axis=0),
                    'paths': tf.concat([val[0]['paths'] + m for val, m in zip(batch_vals,
                                                                              paths_cummax)],
                                       axis=0),
                    'n_links': tf.math.add_n([val[0]['n_links'] for val in batch_vals]),
                    'n_paths': tf.math.add_n([val[0]['n_paths'] for val in batch_vals]),
                    'n_total': tf.math.add_n([val[0]['n_total'] for val in batch_vals])
                    }, tf.concat([val[1] for val in batch_vals], axis=0))

    return tensors


def tfrecord_input_fn(filenames, hparams, shuffle_buf=1000, target='delay'):

    files = tf.data.Dataset.from_tensor_slices(filenames)
    files = files.shuffle(len(filenames))

    # TODO constant 4 should be externalised
    dataset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4))

    if shuffle_buf:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(shuffle_buf))

    # TODO constants 2 and 10 should be externalised
    dataset = dataset.map(lambda buf: parse(buf, target), num_parallel_calls=2)
    dataset = dataset.prefetch(10)

    itrtr = dataset.make_one_shot_iterator()
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
    print('******** in model_fn ******************', file=sys.stderr)

    model = RouteNetModel(params)
    model.build()

    # TODO why this function at all?
    def predict_fn(features):
        preds = model(features, training=mode == tf.estimator.ModeKeys.TRAIN)
        return preds

    predictions = predict_fn(features)

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

    # TODO constants 82000 and 0.8 should be externalised.
    decayed_lr = tf.train.exponential_decay(params.learning_rate,
                                            tf.train.get_global_step(),
                                            82000,
                                            0.8,
                                            staircase=True)

    optimizer = tf.train.AdamOptimizer(decayed_lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grad_var_pairs, global_step=tf.train.get_global_step())

    logging_hook = tf.train.LoggingTensorHook(
        {"Training loss": loss}, every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=[logging_hook])


def parse(serialized, target='delay'):
    # Target is the name of predicted variable
    # TODO 'traffic' below is bandwidth of traffic transmitted
    with tf.device('/cpu:0'):
        with tf.name_scope('parse'):
            features = tf.io.parse_single_example(serialized,
                                                  features={'traffic': tf.io.VarLenFeature(tf.float32),
                                                            target: tf.io.VarLenFeature(tf.float32),
                                                            'link_capacity': tf.io.VarLenFeature(
                                                                tf.float32),
                                                            'links': tf.io.VarLenFeature(tf.int64),
                                                            'paths': tf.io.VarLenFeature(tf.int64),
                                                            'sequences': tf.io.VarLenFeature(tf.int64),
                                                            'n_links': tf.io.FixedLenFeature([],
                                                                                             tf.int64),
                                                            'n_paths': tf.io.FixedLenFeature([],
                                                                                             tf.int64),
                                                            'n_total': tf.io.FixedLenFeature([],
                                                                                             tf.int64)})

            for feature in ['traffic', target, 'link_capacity', 'links', 'paths', 'sequences']:
                # TODO This is a form of normalisation, but why these values? Factor into
                # discrete functions.
                features[feature] = tf.sparse.to_dense(features[feature])
                # TODO create switch for delay normalisation
                if feature == 'delay':
                    features[feature] = (features[feature] - 0.37) / 0.54
                if feature == 'traffic':
                    features[feature] = (features[feature] - 0.17) / 0.13
                if feature == 'link_capacity':
                    features[feature] = (features[feature] - 25.0) / 40.0

    return {k: v for k, v in features.items() if k is not target}, features[target]


def read_dataset(filename, target='delay'):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(lambda buf: parse(buf, target=target))
    dataset = dataset.batch(1)

    return dataset
