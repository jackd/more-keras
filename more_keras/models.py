from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
from more_keras.layers import BatchNormalization
from more_keras.layers import Dense
from more_keras.layers import Dropout


@gin.configurable(module='mk.models')
def mlp(units,
        dense_impl=Dense,
        activation='relu',
        batch_norm_impl=BatchNormalization,
        dropout_impl=None,
        final_units=None,
        activate_first=False):
    """
    Get a sequential model.

    Each block is:
        dense -> batch_norm -> activation -> dropout
    though any/all can be missing except for dense.

    Args:
        units: int or iterable of pythons ints, number of units in each dense
            layer.
        activation: activation used in dense layers after batch norm. Should be
            valid to parse to `tf.keras.layers.Activation` if not `None`.
        dense_impl: dense layer implementation. Should map `units -> layer`.
        batch_norm_impl: batch normalization implementation or None.
            Should map `() -> layer`.
        dropout_impl: optional dropout implementation or None.
            Should map `() -> layer`.
        final_units: if not None, a final `dense_impl` is added at the end
            without any batchnorm/activation/dropout.
        activate_first: if True, batch norm/activation/dropout is potentially
            applied before the first layer.

    Returns:
        `tf.keras.models.Sequential` instance. Note there is no input size,
        so this will not be built until called/built manually using `build`.
    """
    layers = []

    def activate():
        if batch_norm_impl is not None:
            layers.append(batch_norm_impl())
        if activation is not None:
            layers.append(tf.keras.layers.Activation(activation))
        if dropout_impl is not None:
            layers.append(dropout_impl())

    if activate_first:
        activate()

    if isinstance(units, int):
        units = units,
    for u in units:
        layers.append(dense_impl(u))
        activate()
    if final_units is not None:
        layers.append(dense_impl(final_units, activation=None))
    return tf.keras.models.Sequential(layers)
