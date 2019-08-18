from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import functools
import os
from more_keras import callbacks as cb
from more_keras.layers import VariableMomentumBatchNormalization
import more_keras.schedules as sched

from more_keras.framework.problems.tfds import TfdsProblem
from more_keras.framework.pipelines import Pipeline


def val_map_fn(image, labels):
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    return image, labels


def train_map_fn(image, labels, saturation_delta=0.1, hue_max_delta=0.1):
    image = tf.cast(image, tf.float32)
    image = tf.image.random_saturation(image, 1 - saturation_delta,
                                       1 + saturation_delta)
    image = tf.image.random_hue(image, max_delta=hue_max_delta)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
    return image, labels


def get_cifar_model(input_spec,
                    output_spec,
                    training=None,
                    conv_filters=(16, 32),
                    dense_units=(),
                    activation='relu'):
    num_classes = output_spec.shape[-1]
    inputs = tf.keras.layers.Input(shape=input_spec.shape,
                                   dtype=input_spec.dtype)
    x = inputs
    for f in conv_filters:
        x = tf.keras.layers.Conv2D(f, 3)(x)
        x = VariableMomentumBatchNormalization()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Flatten()(x)
    for u in dense_units:
        x = tf.keras.layers.Dense(u)(x)
        x = VariableMomentumBatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
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
        ])
    cb.aggregator.append(updater)
    return tf.keras.Model(inputs=inputs, outputs=logits)


problem = TfdsProblem('cifar100',
                      tf.keras.losses.SparseCategoricalCrossentropy(), [
                          tf.keras.metrics.SparseCategoricalAccuracy(),
                          tf.keras.metrics.SparseCategoricalCrossentropy()
                      ],
                      split_map={'validation': 'test'})

chkpt_dir = os.path.join(os.path.dirname(__file__), 'model')

input_spec = tf.keras.layers.InputSpec(shape=problem.input_spec.shape,
                                       dtype=tf.float32)

batch_size = 32
validation_pipeline = Pipeline(batch_size,
                               repeats=None,
                               map_fn=val_map_fn,
                               output_spec=input_spec)
train_pipeline = Pipeline(batch_size,
                          repeats=None,
                          shuffle_buffer=1024,
                          map_fn=train_map_fn,
                          output_spec=input_spec)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=sched.ExponentialDecayTowards(
        1e-3, 1000, 0.5, clip_value=1e-5))
