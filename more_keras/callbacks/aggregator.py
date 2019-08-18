from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from more_keras.collections import typed_sequence
import contextlib


class Aggregator(object):
    _stack = []

    def __init__(self):
        self._callbacks = typed_sequence(tf.keras.callbacks.Callback)

    def __enter__(self):
        Aggregator._stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        popped = Aggregator._stack.pop()
        assert (popped is self)

    @property
    def callbacks(self):
        return self._callbacks

    @classmethod
    def current(cls):
        if not cls._stack:
            raise RuntimeError(
                'callbacks Aggregator stack empty. Use this method within a'
                'context block, `with {}() as cache: ...`'.format(cls.__name__))
        return cls._stack[-1]

    def append(self, callback):
        self._callbacks.append(callback)


current = Aggregator.current


def append(callback):
    return current().append(callback)
