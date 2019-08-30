from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin
from more_keras.layers.batch_norm import VariableMomentumBatchNormalization
from more_keras.layers.batch_norm import ConsistentBatchNormalization
from more_keras.layers import utils

# Configurable versions of base layers to be used from code.
BatchNormalization = gin.external_configurable(
    tf.keras.layers.BatchNormalization, module='mk.layers')
Convolution1D = gin.external_configurable(tf.keras.layers.Convolution1D,
                                          module='mk.layers')
Convolution2D = gin.external_configurable(tf.keras.layers.Convolution2D,
                                          module='mk.layers')
Convolution3D = gin.external_configurable(tf.keras.layers.Convolution3D,
                                          module='mk.layers')
Dense = gin.external_configurable(tf.keras.layers.Dense, module='mk.layers')
Dropout = gin.external_configurable(tf.keras.layers.Dropout, module='mk.layers')

# clean up namespace
del gin, tf
