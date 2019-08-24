from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import collections
import functools
from more_keras.utils import UpdateFrequency


def py_function_lambda(args, func, Tout):
    return tf.py_function(func, args, Tout=Tout)


class Cache(tf.keras.callbacks.Callback, collections.Mapping):  # pylint: disable=no-member
    """
    Workaround for using model parameters in dataset map functions.

    Datasets can't use map functions that involve variables. We get around this
    by using no-argument `tf.py_function` outputs that look up values stored
    in a cache. At the end of each batch/epoch, we update that cache using
    `tf.keras.backend.get_value`.

    Not supported in eager mode (you should be able to just use the tensor).
    """
    _stack = []

    @classmethod
    def current(cls):
        if not cls._stack:
            raise RuntimeError(
                '`{name}` stack empty. Use this method within a context '
                'block, e.g. `with {name}() as cache: ...`'.format(
                    name=cls.__name__))
        return cls._stack[-1]

    def __init__(self, update_freq=UpdateFrequency.BATCH):
        self._dirty_tensors = {}  # dict mapping tensors to py_function tensors
        self._np_values = {}
        UpdateFrequency.validate(update_freq)
        self._update_freq = update_freq

    def __enter__(self):
        Cache._stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        out = Cache._stack.pop()
        assert (out is self)

    def on_train_batch_end(self, batch, logs=None):
        if self._update_freq == UpdateFrequency.BATCH:
            self.update_cache()

    def on_epoch_end(self, epoch, logs=None):
        if self._update_freq == UpdateFrequency.EPOCH:
            self.update_cache()

    def on_train_begin(self, logs=None):
        self.update_cache()

    def on_test_begin(self, logs=None):
        self.update_cache()

    def on_predict_begin(self, logs=None):
        self.update_cache()

    def update_cache(self):
        self._np_values.update({
            k: tf.keras.backend.get_value(k)
            for k in self._dirty_tensors.keys()
        })

    def _get_value(self, tensor):
        return self._np_values[tensor]

    def __getitem__(self, tensor):
        if tf.executing_eagerly():
            raise RuntimeError('Cannot use `Cache` values in eager mode.')
        out = self._dirty_tensors.get(tensor, None)
        if out is None:
            fn = functools.partial(self._get_value, tensor=tensor)
            out = tf.keras.layers.Lambda(py_function_lambda,
                                         arguments=dict(func=fn,
                                                        Tout=tensor.dtype))([])
            out.set_shape(tensor.shape)
            self._dirty_tensors[tensor] = out

        return out

    def get_cached(self, tensor, value):
        self._np_values.setdefault(tensor, value)
        return self[tensor]

    def __iter__(self):
        return iter(self._dirty_tensors)

    def __len__(self):
        return len(self._dirty_tensors)


current = Cache.current


def get_cached(tensor, default_value):
    return current().get_cached(tensor, default_value)
