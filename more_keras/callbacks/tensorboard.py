from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import collections
import functools
import six
from more_keras.utils import UpdateFrequency
import gin


@gin.configurable(module='mk.callbacks')
class BetterTensorBoard(tf.keras.callbacks.TensorBoard,
                        collections.MutableMapping):  # pylint: disable=no-member
    """
    Base TensorBoard callback makes it difficult to add custom summaries.

    This allows scalars to be added easily. Bring on 2.0...

    Note that by combining adding to logs with the tensorboard callback we
    ensure summary values are only called when they'll be logged.

    Example usage:
    ```python
    with BetterTensorBoard() as tb:
        tb['custom_name'] = my_custom_scalar_tensor

    callbacks.append(tb)
    model.fit(..., callbacks=callbacks)
    """
    _stack = []

    @classmethod
    def current(cls):
        if not cls._stack:
            raise RuntimeError(
                '`{name}` stack empty. Use this method within a context '
                'block, e.g. `with {name}() as {lower}: ...`'.format(
                    name=cls.__name__, lower=cls.__name__.lower()))
        return cls._stack[-1]

    def __init__(self,
                 log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 embeddings_data=None,
                 update_freq='epoch',
                 profile_batch=2):
        self._tensors = {}
        super(BetterTensorBoard, self).__init__(
            log_dir=log_dir,
            histogram_freq=histogram_freq,
            batch_size=batch_size,
            write_graph=write_graph,
            write_grads=write_grads,
            write_images=write_images,
            embeddings_freq=embeddings_freq,
            embeddings_layer_names=embeddings_layer_names,
            embeddings_metadata=embeddings_metadata,
            embeddings_data=embeddings_data,
            update_freq=update_freq,
            profile_batch=profile_batch,
        )

    def _write_custom_summaries(self, step, logs=None):
        logs = self._update_logs(logs)
        return super(BetterTensorBoard,
                     self)._write_custom_summaries(step, logs)

    def __enter__(self):
        BetterTensorBoard._stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        out = BetterTensorBoard._stack.pop()
        assert (out is self)

    def _update_logs(self, logs=None):
        if logs is None:
            logs = {}
        np_values = {
            k: tf.keras.backend.get_value(v) for k, v in self._tensors.items()
        }
        for k in np_values:
            if k in logs:
                raise ValueError('duplicate key from LogUpdater')
        logs.update(np_values)
        return logs

    def __setitem__(self, key, tensor):
        if not isinstance(key, six.string_types):
            raise KeyError('Only string keys allowed')
        if key in self._tensors and self[key] != tensor:
            raise KeyError(
                'key {} already present with different value'.format(key))
        self._tensors[key] = tensor

    def __getitem__(self, key):
        return self._tensors[key]

    def __iter__(self):
        return iter(self._tensors)

    def __len__(self):
        return len(self._tensors)

    def __delitem__(self, key):
        del self._tensors[key]

    def __contains__(self, key):
        return key in self._tensors


current = BetterTensorBoard.current


def add_custom_scalar(key, tensor):
    current()[key] = tensor
