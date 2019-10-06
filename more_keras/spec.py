from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def to_spec(tensor):
    if tensor is None:
        return None
    elif isinstance(tensor, tf.RaggedTensor):
        return tf.RaggedTensorSpec.from_value(tensor)
    elif isinstance(tensor, tf.Tensor):
        return tf.TensorSpec.from_tensor(tensor)
    else:
        raise TypeError(
            'Expected Tensor or RaggedTensor, got {}'.format(tensor))


def from_spec(spec):
    if spec is None:
        return None
    elif hasattr(tf, 'RaggedTensorSpec') and isinstance(spec,
                                                        tf.RaggedTensorSpec):
        (values,
         *nested_row_splits) = [from_spec(c) for c in spec._to_components]
        return tf.RaggedTensor.from_nested_row_splits(values, nested_row_splits)
    elif isinstance(spec, tf.TensorSpec):
        if spec.shape[0] is None:
            return tf.keras.layers.InputSpec(shape=spec.shape[1:],
                                             dtype=spec.dtype)
        else:
            return tf.keras.layers.InputSpec(shape=spec.shape,
                                             dtype=spec.dtype)[0]


def get(identifier):
    if identifier is None:
        return identifier
    elif isinstance(identifier, tf.TypeSpec):
        return identifier
    elif isinstance(identifier, dict):
        if 'class_name' in identifier:
            class_name = identifier['class_name']
            config = identifier['config']
            return getattr(tf, class_name)(**config)
        else:
            return tf.nest.map_structure(get, identifier)
    elif isinstance(identifier, (list, tuple)):
        return tf.nest.map_structure(get, identifier)
    else:
        raise TypeError('Unrecognized identifier {}'.format(identifier))


def serialize(spec):
    if spec is None:
        return None
    elif isinstance(spec, tf.TensorSpec):
        return dict(class_name='TensorSpec',
                    config=dict(shape=spec.shape, dtype=str(spec.dtype)[3:]))
    elif hasattr(tf, 'RaggedTensorSpec') and isinstance(spec,
                                                        tf.RaggedTensorSpec):
        return dict(class_name='RaggedTensorSpec',
                    config=dict(shape=spec.shape,
                                dtype=str(spec.dtype)[3:],
                                ragged_rank=spec.ragged_rank,
                                row_splits_dtype=str(spec.dtype)[3:]))
    elif isinstance(spec, (dict, list, tuple)):
        return tf.nest.map_structure(spec, serialize)


def _input(spec):
    ragged = hasattr(tf, 'RaggedTensorSpec') and isinstance(
        spec, tf.RaggedTensorSpec)
    if ragged:
        batch_size = spec._component_specs[1].shape[0]
        if batch_size is not None:
            batch_size = batch_size - 1
        return tf.keras.layers.Input(shape=spec._shape[1:],
                                     dtype=spec._component_specs[0].dtype,
                                     name=spec._component_specs[0].name,
                                     batch_size=batch_size,
                                     ragged=True)
    elif isinstance(spec, tf.TensorSpec):
        return tf.keras.layers.Input(shape=spec.shape[1:],
                                     dtype=spec.dtype,
                                     name=spec.name,
                                     batch_size=spec.shape[0])
    else:
        raise ValueError('Expected a tf.TensorSpec, got {}'.format(spec))


def inputs(spec):
    """
    Get tf.keras.layers.InputSpec corresponding to the provided spec(s).

    Args:
        spec: TensorSpec, RaggedTensorSpec or arbitrarily nested structure of
            them.

    Returns:
        `tf.keras.layers.Input` or the corresponding structure.
    """
    return tf.nest.map_structure(_input, spec)


def element_spec(dataset):
    try:
        return dataset.element_spec
    except AttributeError:
        # 1.14.0 or lower
        return tf.nest.map_structure(
            lambda shape, dtype: tf.TensorSpec(shape, dtype),
            dataset.output_shapes, dataset.output_types)
