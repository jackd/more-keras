from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin
import tensorflow as tf


@gin.configurable(module='mk.callbacks')
class GinConfigSaver(tf.keras.callbacks.Callback):

    def __init__(self, directory, overwrite=True):
        self.directory = os.path.expanduser(os.path.expandvars(directory))
        self.overwrite = overwrite

    def on_train_begin(self, logs=None):
        path = os.path.join(self.directory, 'operative-config.gin')
        if not self.overwrite and tf.io.gfile.exists(path):
            raise RuntimeError(
                'Cannot write gin config - file already exists at {}'.format(
                    path))
        if not tf.io.gfile.isdir(self.directory):
            tf.io.gfile.makedirs(self.directory)

        with tf.io.gfile.GFile(path, 'w') as fp:
            fp.write(gin.operative_config_str())
