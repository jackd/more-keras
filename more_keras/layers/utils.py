"""
tf.keras.layers.Lambda wrappers around more_keras.ops.utils.

Hopefully this can largely be dropped for 2.0.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from tensorflow.python.util import tf_inspect  # pylint: disable=no-name-in-module
import tensorflow as tf
from more_keras.ragged.layers import maybe_ragged_lambda_call
from more_keras.ragged.layers import ragged_lambda
from more_keras.ops import utils as _utils


def _lambda_call_fn(inputs, fn, input_names, extra_kwargs):
    if len(input_names) == 1 and isinstance(inputs, tf.Tensor):
        kwargs = {input_names[0]: inputs}
    else:
        assert (len(inputs) == len(input_names))
        kwargs = {n: i for n, i in zip(input_names, inputs)}
    kwargs.update(extra_kwargs)
    return fn(**kwargs)


def lambda_call(*args, **kwargs):
    """
    Convenience wrapper to `tf.keras.layers.Lambda`.

    Expects `fn` as either 0th arg or as a key in kwargs.

    Example usage: the following will be outputs of serializable.
    ```python
    y = lambda_call(tf.squeeze, x, axis=-1)
    ```
    Equivalent to
    ```python
    y = tf.keras.layers.Lambda(tf.squeeze, arguments=dict(axis=-1))(x)
    ```

    ```python
    def triple_sum(x, y, z):
        return x + y + z

    s3 = lambda_call(triple_sum, x_tensor, y=y_tensor, z=3)
    ```
    Equivalent to
    ```python
    s3 = tf.keras.layers.Lambda(
        triple_sum, arguments=dict(z=3))([x_tensor, y_tensor])
    ```

    """
    if 'fn' in kwargs:
        fn = kwargs.pop('fn')
    else:
        fn = args[0]
        args = args[1:]
    extra_kwargs = {}
    tensor_kwargs = {}
    names = tf_inspect.getargspec(fn).args
    for n, a in zip(names, args):
        if n in kwargs:
            raise ValueError('argument name %s provided twice' % n)
        kwargs[n] = a
    for k, v in kwargs.items():
        if isinstance(v, (tf.Tensor, tf.Variable)):
            tensor_kwargs[k] = v
        else:
            # check json serializability?
            if isinstance(v, tf.DType):
                v = repr(v)[3:]
            extra_kwargs[k] = v

    names = tuple(tensor_kwargs.keys())
    values = [tensor_kwargs[k] for k in names]
    layer = tf.keras.layers.Lambda(_lambda_call_fn,
                                   arguments=dict(fn=fn,
                                                  input_names=names,
                                                  extra_kwargs=extra_kwargs))
    return layer(values)


def lambda_wrapper(fn):
    return functools.partial(lambda_call, fn=fn)


row_lengths_to_splits = lambda_wrapper(_utils.row_lengths_to_splits)
diff = lambda_wrapper(_utils.diff)
cast = lambda_wrapper(tf.cast)
squeeze = lambda_wrapper(tf.squeeze)
size = lambda_wrapper(tf.size)
divide = lambda_wrapper(tf.math.divide)
reciprocal = lambda_wrapper(tf.math.reciprocal)


def flatten_final_dims(tensor, n=2):
    return maybe_ragged_lambda_call(_utils.flatten_final_dims,
                                    tensor,
                                    arguments=dict(n=n))


def flatten_leading_dims(tensor, n=2):
    return maybe_ragged_lambda_call(_utils.flatten_leading_dims,
                                    tensor,
                                    arguments=dict(n=n))


def reshape_final_dim(tensor, final_dims):
    return maybe_ragged_lambda_call(_utils.reshape_final_dim,
                                    tensor,
                                    arguments=dict(final_dims=final_dims))


def merge_features(node_features, edge_features):
    return maybe_ragged_lambda_call(_utils.merge_features,
                                    [node_features, edge_features])


def get_row_offsets(tensor):
    return maybe_ragged_lambda_call(_utils.get_row_offsets, tensor)


def reduce_sum(tensor, axis=None, keepdims=False):
    return maybe_ragged_lambda_call(tf.reduce_sum,
                                    tensor,
                                    arguments=dict(axis=axis,
                                                   keepdims=keepdims))


def _row_lengths(rt):
    return rt.row_lengths()


def row_lengths(row_splits):
    if isinstance(row_splits, tf.RaggedTensor):
        return ragged_lambda(_row_lengths)(row_splits)
    else:
        return diff(row_splits)


def _gather(args, **kwargs):
    return tf.gather(*args, **kwargs)


def gather(params, indices, **kwargs):
    return maybe_ragged_lambda_call(_gather, [params, indices],
                                    arguments=kwargs)


def _map_gather(args, **kwargs):
    return _utils.map_gather(*args, **kwargs)


def map_gather(params, indices, **kwargs):
    return maybe_ragged_lambda_call(_map_gather, [params, indices],
                                    arguments=kwargs)


def _apply_row_offset(args):
    indices, offset = args
    return _utils.apply_row_offset(indices, offset)


def apply_row_offset(indices, offset):
    return maybe_ragged_lambda_call(_apply_row_offset, [indices, offset])


def _leading_dim(x, dtype=tf.int64):
    return tf.shape(x, out_type=dtype)[0]


leading_dim = lambda_wrapper(_leading_dim)
