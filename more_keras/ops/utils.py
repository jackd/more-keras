from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import numpy as np
import tensorflow as tf


def outer(node_features, edge_features):
    """
    Outer product along final dimension.

    Equivalent to `tf.einsum('ij,ik->ijk', node_features, edge_featues)` when
    inputs are both rank 2.

    Args:
        node_features: [B*, n] float32 array of features at each node
        edge_features: [B*, e] float32 array of features at each edge

    Returns:
        [B, n, e] float array of the product.

    *: Inputs may be different shapes, but shapes up to the final dimension
        must be broadcastable.
    """
    node_features = tf.expand_dims(node_features, axis=-1)
    edge_features = tf.expand_dims(edge_features, axis=-2)
    return node_features * edge_features


def _flatten_final_dims(tensor, n=2):
    leading, trailing = tf.split(tf.shape(tensor), [tensor.shape.ndims - n, n])
    trailing_actual = tensor.shape[-n:]
    if trailing_actual.is_fully_defined():
        final = trailing_actual.num_elements()
    else:
        final = tf.reduce_prod(trailing)
    return tf.reshape(tensor, tf.concat([leading, [final]], axis=0))


def flatten_final_dims(tensor, n=2):
    """
    Reshape `tensor` by combining `n` final dimensions.

    e.g.
    ```python
    x = tf.zeros((5, 6, 3, 2))
    flat = flatten_final_dims(x, n=3)
    print(flat.shape)  # (5, 6*3*2)
    ```
    """
    if n == 1:
        return tensor
    fn = functools.partial(_flatten_final_dims, n=n)
    if isinstance(tensor, tf.RaggedTensor):
        return tf.ragged.map_flat_values(fn, tensor)
    else:
        return fn(tensor)


def _reshape_final_dim(tensor, final_dims):
    assert (isinstance(final_dims, tuple) and
            all(isinstance(f, int) for f in final_dims))
    assert (np.prod(final_dims) == tensor.shape[-1].value)
    leading_dims = tf.shape(tensor)[:-1]
    shape = tf.concat([leading_dims, final_dims], axis=0)
    return tf.reshape(tensor, shape)


def reshape_final_dim(tensor, final_dims):
    fn = functools.partial(_reshape_final_dim, final_dims=final_dims)
    if isinstance(tensor, tf.RaggedTensor):
        return tf.ragged.map_flat_values(fn, tensor)
    else:
        return fn(tensor)


def flatten_leading_dims(tensor, n=2):
    """Reshape `tensor` by combining `n` leading dimensions."""
    if n == 1:
        return tensor
    if isinstance(tensor, tf.RaggedTensor):
        return flatten_leading_dims(tensor.values, n=n - 1)
    else:
        leading, trailing = tf.split(tf.shape(tensor),
                                     [n, tensor.shape.ndims - n])
        if tensor.shape[:n].is_fully_defined():
            initial = tensor.shape[:n].num_elements()
        else:
            initial = tf.reduce_prod(leading)
        if tensor.shape[n:].is_fully_defined():
            trailing = tensor.shape[n:]
        return tf.reshape(tensor, tf.concat([[initial], trailing], axis=0))


def get_row_offsets(tensor, dtype=tf.int64):
    if isinstance(tensor, tf.RaggedTensor):
        return tensor.row_starts()
    else:
        shape = tf.shape(tensor, out_type=dtype)
        batch_size = shape[0]
        stride = shape[1]
        return tf.range(batch_size) * stride


def map_gather(params, indices):
    """
    Equivalent to tf.map(tf.gather, (params, indices)).

    Implementation is based on modifying indices and does not actually use
    `tf.map`.
    """
    row_offset = get_row_offsets(params, dtype=indices.dtype)
    indices = apply_row_offset(indices, row_offset)
    params = flatten_leading_dims(params, n=2)
    out = tf.gather(params, indices)
    return out


def apply_row_offset(indices, offset):
    return indices + tf.reshape(offset,
                                (-1,) + (1,) * (indices.shape.ndims - 1))


def row_lengths_to_splits(row_lengths):
    return tf.pad(tf.cumsum(row_lengths), [[1, 0]])


def _diff(tensor, axis=-1):
    top = tf.split(tensor, [1, -1], axis=axis)[1]
    bottom = tf.split(tensor, [-1, 1], axis=axis)[0]
    return top - bottom


def diff(tensor, n=1, axis=-1):
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer, got %s' % str(n))
    for _ in range(n):
        tensor = _diff(tensor, axis=axis)
    return tensor
