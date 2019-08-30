# Included to help examples. See more-keras/examples/cifar100.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import functools
import os
import gin
from more_keras import callbacks as cb
from more_keras.layers import VariableMomentumBatchNormalization
import more_keras.schedules as sched

from more_keras.framework.problems.tfds import TfdsProblem
from more_keras.framework.pipelines import Pipeline
from more_keras import spec


@gin.configurable
def val_map_fn(image, labels):
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return image, labels


@gin.configurable
def train_map_fn(image, labels, saturation_delta=0.1, hue_max_delta=0.1):
    image = tf.cast(image, tf.float32)
    # image = tf.image.random_saturation(image, 1 - saturation_delta,
    #                                    1 + saturation_delta)
    # image = tf.image.random_hue(image, max_delta=hue_max_delta)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    return image, labels


@gin.configurable(blacklist=['input_spec', 'output_spec'])
def get_model(input_spec,
              output_spec,
              conv_filters=(16, 32),
              dense_units=(),
              activation='relu'):
    inputs = spec.inputs(input_spec)
    x = inputs
    for f in conv_filters:
        x = tf.keras.layers.Conv2D(f, 3)(x)
        x = VariableMomentumBatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Flatten()(x)
    for u in dense_units:
        x = tf.keras.layers.Dense(u)(x)
        x = VariableMomentumBatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)

    num_classes = output_spec.shape[-1]
    logits = tf.keras.layers.Dense(num_classes)(x)

    updater = cb.schedule_updater.ScheduleUpdater(
        schedule=functools.partial(sched.exponential_decay_towards,
                                   initial_value=0.5,
                                   decay_steps=1,
                                   decay_rate=0.5,
                                   asymptote=1.0,
                                   clip_value=0.99,
                                   impl=np),
        variables_func=lambda model: [
            l.momentum
            for l in model.layers
            if isinstance(l, VariableMomentumBatchNormalization)
        ],
        logs_key='batch_norm_momentum')
    cb.aggregator.append(updater)
    return tf.keras.Model(inputs=inputs, outputs=logits)


@gin.configurable
def get_problem():
    return TfdsProblem(
        'cifar100',
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        [
            tf.keras.metrics.SparseCategoricalAccuracy(),
            # tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
        ],
        split_map={'validation': 'test'})
