from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


class UpdateFrequency(object):
    BATCH = 'batch'
    EPOCH = 'epoch'

    @classmethod
    def validate(cls, freq):
        if freq not in (cls.BATCH, cls.EPOCH):
            raise ValueError(
                'Invalid frequency "{}" - must be one of {}'.format(
                    freq, (cls.BATCH, cls.EPOCH)))


def compute_output_spec(input_spec, map_fn):
    with tf.Graph().as_default():  # pylint: disable=not-context-manager
        inp = tf.keras.layers.Input(shape=input_spec.shape,
                                    dtype=input_spec.dtype)[0]
        out = map_fn(inp)
        return tf.keras.layers.InputSpec(dtype=out.dtype, shape=out.shape)
