from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from more_keras import spec


def assert_compatible(spec, value):
    if not spec.is_compatible_with(value):
        raise ValueError('spec {} not compatible with value {}'.format(
            spec, value))


def ragged_tensor_value(components):
    if len(components) == 2:
        return tf.ragged.RaggedTensorValue(*components)
    else:
        return tf.ragged.RaggedTensorValue(
            ragged_tensor_value([components[0], *components[2:]]),
            components[1])


class FlatPacker(object):

    def __init__(self, element_spec):
        self._elements = element_spec
        self._flat_elements = tf.nest.flatten(self._elements)
        self._components = [
            getattr(x, '_component_specs', x) for x in self._flat_elements
        ]
        self._flat_components = tf.nest.flatten(self._components)
        for c in self._flat_components:
            if c.shape.as_list().count(None) > 1:
                raise ValueError(
                    'Cannot pack an element with more than 1 unknown dimension')
        self._num_flat = len(self._flat_components)
        self._dtype = tf.float64 if any(
            c.dtype == tf.float64
            for c in self._flat_components) else tf.float32

    def pack(self, elements):
        tf.nest.map_structure(assert_compatible, self._elements, elements)
        components = tf.nest.map_structure(
            lambda x: x._to_components()
            if hasattr(x, '_to_components') else x, elements)
        flat_components = tf.nest.flatten(components)
        flat_components = [tf.reshape(c, (-1,)) for c in flat_components]
        sizes = tf.stack([tf.size(c) for c in flat_components], axis=0)
        all_data = [sizes] + flat_components
        all_data = [tf.cast(d, self._dtype) for d in all_data]
        return tf.concat(all_data, axis=0)

    def unpack(self, flat_values):
        sizes, rest = tf.split(flat_values, [self._num_flat, -1], axis=0)
        sizes = tf.cast(sizes, dtype=tf.int64)
        flat_components = tf.split(rest, sizes, axis=0)
        flat_components = tf.nest.map_structure(
            lambda x, sp: tf.cast(
                tf.reshape(x, [-1 if d is None else d
                               for d in sp.shape]), sp.dtype), flat_components,
            self._flat_components)
        components = tf.nest.pack_sequence_as(self._components, flat_components)
        components = [
            spec._from_components(t)
            for spec, t in zip(self._flat_elements, components)
        ]
        components = tf.nest.pack_sequence_as(self._elements, components)
        return components

    def unpack_numpy(self, flat_values):
        import numpy as np
        if not all(
                isinstance(spec, (tf.RaggedTensorSpec, tf.TensorSpec))
                for spec in self._flat_elements):
            raise NotImplementedError()
        sizes, rest = np.split(flat_values, [self._num_flat, -1], axis=0)  # pylint: disable=unbalanced-tuple-unpacking
        indices = np.cumsum(sizes)
        flat_components = np.split(rest, indices)
        flat_components = tf.nest.map_structure(
            lambda x, sp: np.reshape(
                x, [-1 if d is None else d for d in sp.shape]).astype(sp.dtype),
            flat_components, self._flat_components)
        components = tf.nest.pack_sequence_as(self._components, flat_components)
        components = [
            ragged_tensor_value(c)
            if isinstance(spec, tf.RaggedTensorSpec) else c
            for spec, c in zip(self._flat_elements, components)
        ]
        return tf.nest.pack_sequenc_as(self._elements, components)
