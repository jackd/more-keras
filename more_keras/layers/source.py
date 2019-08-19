from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


class ConstantSource(tf.keras.layers.Layer):

    def __init__(self, value, dtype, **kwargs):
        self._value = value
        self._dtype = dtype
        super(ConstantSource, self).__init__(**kwargs)

    def build(self, input_shapes):
        self._tensor = tf.constant(self._value, dtype=self._dtype)
        super(ConstantSource, self).build(input_shapes)

    def call(self, inputs):
        return self._tensor

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self._value.shape)


def constant(x, dtype=None):
    out = ConstantSource(x, dtype=dtype)([])
    return out


class VariableSource(tf.keras.layers.Layer):

    def __init__(self, shape, dtype, initializer, **kwargs):
        self._shape = tf.TensorShape(shape)
        self._initializer = initializer
        super(VariableSource, self).__init__(dtype=dtype, **kwargs)

    def build(self, input_shapes):
        self._weight = self.add_weight('weight',
                                       self._shape,
                                       dtype=self._dtype,
                                       initializer=self._initializer)
        super(VariableSource, self).build(input_shapes)

    def call(self, inputs):
        return self._weight

    def compute_output_shape(self, input_shape):
        return self._shape


def variable(shape, dtype, initializer, name=None):
    return VariableSource(shape, dtype, initializer, name=name)([])
