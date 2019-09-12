from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf


@gin.configurable('mk.metrics')
class ProbMeanIoU(tf.keras.metrics.MeanIoU):
    """tf.keras.metrics.MeanIoU wrapper that takes probabilities/logits."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super(ProbMeanIoU, self).update_state(y_true, y_pred,
                                                     sample_weight)
