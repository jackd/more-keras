from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        'No tensorflow installation found. `more_keras` does not '
        'automatically install tensorflow. Please install manaully.')
import distutils.version
tf.compat.v1.enable_v2_tensorshape()

tf_version = distutils.version.LooseVersion(tf.__version__)
is_v1 = tf_version.version[0] == 1
is_v2 = tf_version.version[0] == 2

if not (is_v1 or is_v2):
    raise ImportError(
        'Detected version of tensorflow %s not compatible with more_keras - '
        'only versions 1 and 2 supported' % (tf.__version__))

is_v1_13 = is_v1 and tf_version.version[1] == 13
