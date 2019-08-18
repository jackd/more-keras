from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging
import tensorflow as tf


class SessionOptions(object):

    def __init__(self, allow_growth=True, visible_devices=None, eager=False):
        self.allow_growth = allow_growth
        self.visible_devices = visible_devices
        self.eager = eager

    def configure_session(self):
        visible_devices = self.visible_devices
        if visible_devices is not None:
            if isinstance(visible_devices, (list, tuple)):
                visible_devices = (',').join(str(d) for d in visible_devices)
            elif isinstance(visible_devices, int):
                visible_devices = str(visible_devices)
            logging.info(
                'Setting CUDA_VISIBLE_DEVICES={}'.format(visible_devices))
            os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices

        import tensorflow as tf
        if self.allow_growth is not None:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True  # pylint: disable=no-member
        else:
            config = None
        if self.eager:
            tf.compat.v1.enable_eager_execution(config=config)
        elif config is not None:
            tf.keras.backend.set_session(tf.compat.v1.Session(config=config))

    def get_config(self):
        return dict(allow_growth=self.allow_growth,
                    visible_devices=self.visible_devices,
                    eager=self.eager)

    @classmethod
    def from_config(cls, config):
        return SessionOptions(**config)
