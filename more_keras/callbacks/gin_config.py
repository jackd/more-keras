from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin
import tensorflow as tf


class GinConfigSaver(tf.keras.callbacks.Callback):

    def __init__(self, path, overwrite=True):
        path = os.path.expanduser(os.path.expandvars(path))
        self.path = path
        self.overwrite = overwrite

    def on_train_begin(self, logs=None):
        if not self.overwrite and tf.io.gfile.exists(self.path):
            raise RuntimeError(
                'Cannot write gin config - file already exists at {}'.format(
                    self.path))
        with tf.io.gfile.GFile(self.path, 'w') as fp:
            fp.write(gin.operative_config_str())
