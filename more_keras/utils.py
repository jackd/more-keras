from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import gin
from absl import logging

DEFAULT = '__default__'


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
    import tensorflow as tf
    with tf.Graph().as_default():  # pylint: disable=not-context-manager
        inp = tf.keras.layers.Input(shape=input_spec.shape,
                                    dtype=input_spec.dtype)[0]
        out = map_fn(inp)
        return tf.keras.layers.InputSpec(dtype=out.dtype, shape=out.shape)


@gin.configurable(module='mk.utils')
def identity(x):
    return x


@gin.configurable(module='mk.utils')
def ray_init(redis_address=DEFAULT,
             num_cpus=None,
             num_gpus=None,
             local_mode=False,
             **kwargs):
    import ray
    if redis_address == DEFAULT:
        redis_address = os.environ.get('REDIS_ADDRESS')
    return ray.init(redis_address,
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    local_mode=local_mode,
                    **kwargs)


@gin.configurable(module='mk.utils')
def proc(title='more_keras'):
    if title is not None:
        try:
            import setproctitle
            setproctitle.setproctitle(title)
        except ImportError:
            logging.warning(
                'Failed to import setproctitle - cannot change title.')
