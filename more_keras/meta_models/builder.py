"""
Provides tools for building meta-networks.

Meta-network building allows code associated with conceptual layers to be
written in one place, even when it involves pre-batch, post-batch and main
network operations. This is useful when you have networks with per-layer
preprocessing.

For example, consider the following code.

```python

class MyLayerBuilder(object):
    def prebatch_map(self, example_data):
        ...

    def network_fn(self, batch_data, batched_preprocessed_data):
        ...


layer1_builder = MyLayerBuilder()
layer1_builder = MyLayerBuilder()


def prebatch_preprocess(original_inputs):
    x = original_inputs
    x, layer1_prep_inputs = layer_builder1.prebatch_map(x)
    _, layer1_prep_inputs = layer_builder2.prebatch_map(x)
    return original_inputs, layer1_prep_inputs, layer2_prep_inputs


def get_model(inputs):
    x, layer1_prep_inputs, layer2_prep_inputs = inputs
    x = layer1_builder.network_fn(x, layer1_prep_inputs)
    x = layer2_builder.network_fn(x, layer2_prep_inputs)
    return tf.keras.models.Model(inputs=inputs, outputs=x)


dataset = get_original_dataset(...)
dataset = dataset.repeat().shuffle(buffer_size).map(
    lambda inputs, labels: prebatch_preprocess(inputs), labels).batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)

inputs, labels = tf.nest.map_structure(
    lambda s, d: tf.keras.layers.Input(shape=s, dtype=d),
    dataset.output_shapes, dataset.output_types)

model = get_model(inputs)
model.compile(...)
model.fit(dataset, ...)
```

This can be done using this module as follows.
```python
from more_keras.meta_models import builder as b
dataset = get_original_dataset(...)

inputs, labels = b.prebatch_inputs_from(dataset)
x = inputs
x, layer1_prep_inputs = layer1_builder.prebatch_map(x)
_, layer2_prep_inputs = layer2_builder.prebatch_map(x)

batched_layer1_prep_inputs = b.as_model_inputs(layer1_prep_inputs)
batched_layer2_prep_inputs = b.as_model_inputs(layer2_prep_inputs)
batched_inputs = b.as_model_input(inputs)

batched_x = batched_inputs
batched_x = layer1_builder.network_fn(batched_x, batched_layer1_prep_inputs)
batched_x = layer2_builder.network_fn(batched_x, batched_layer2_prep_inputs)

model = b.model(batched_x)
preprocessor = b.preprocessor(b.batched(labels))

dataset = preprocessor.map_and_batch(
    dataset.repeat().shuffle(buffer_size)).prefetch(
        tf.data.experimental.AUTOTUNE)

model.compile(...
model.fit(dataset, ...)
```
While the benefit may not be clear in this example, keep in mind this is a
simple sequential 2-layer model. The cost of developing, maintaining and
coordinating separate preprocessing and network functions grows rapidly with
model complexity. # TODO: add example

Post-batch preprocessing is also supported by using separate `b.batched` and
`b.as_model_input` calls (`b.as_batched_model_input` is a simple wrapper around
both of these).

Under the hood, this is done by keeping track of different sets of inputs and
outputs which go on to form prebatch, postbatch and learned keras models. For
example, `batched_x = b.batched(x)` marks `x` as an output of the prebatch map
model and `batched_x` as an input to the postbatch map model, while
`model_z = b.as_model_input(batched_y)` will mark `model_x` as a learned model
input and `batched_y` as an output of the postbatch map model.

If you wish to build multiple meta-networks separately, you can create
`MetaNetworkBuilder`s and use context blocks.

```python
from more_keras.meta_models import builder as b
first_builder = b.MetaNetworkBuilder()

with first_builder:
    ...
    second_builder = b.MetaNetworkBuilder()
    with second_builder:
        # b.foo() redirects to second_builder.foo()
        b1_input = b.prebatch_input(...)
        b0_input = first_builder.prebatch_input(...)
        ...

m0 = b0.model(m0_outputs)
m1 = b1.model(m1_outputs)
p0 = b0.preprocessor()
p1 = b1.preprocessor()
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from more_keras.layers import utils as layer_utils
from more_keras.meta_models import utils as meta_utils
from more_keras.meta_models.preprocessor import Preprocessor
from more_keras.ragged import layers as ragged_layers
# from tensorflow.python.framework import composite_tensor as _comp


class Marks(object):
    """
    Different marks for tensors in a meta-network.

    Each tensor in a meta-network should be part of only 1 of prebatch,
    postbatch or learned (model) networks. If, for example, you wish for a
    batched network tensor to be used as part of the model, create a new tensor
    using `b.as_model_input(batched_tensor)`.
    """
    PREBATCH = 0
    BATCHED = 1
    MODEL = 2
    LABEL = 3

    _strings = ('prebatch', 'batched', 'model', 'label')

    @classmethod
    def to_string(cls, mark):
        return cls._strings[mark]


class MetaNetworkBuilder(object):
    """See `help(more_keras.meta_models.builder)`."""

    _stack = []

    @classmethod
    def current(cls):
        if not cls._stack:
            raise RuntimeError(
                '`{name}` stack empty. Use this method within a context '
                'block, e.g. `with {name}() as cache: ...`'.format(
                    name=cls.__name__))
        return cls._stack[-1]

    def __init__(self):
        self._prebatch_inputs = []
        self._prebatch_outputs = []
        self._batched_inputs = []
        self._batched_feature_outputs = []
        self._model_inputs = []

        self._batched_inputs_dict = {}
        self._model_inputs_dict = {}

        self._marks = {}

    def get_mark(self, tensor):
        """See `more_keras.meta_models.builder.Marks`."""
        return self._marks.get(tensor)

    def __enter__(self):
        MetaNetworkBuilder._stack.append(self)
        return self

    def __exit__(self, *args, **kwargs):
        out = MetaNetworkBuilder._stack.pop()
        if out is not self:
            raise RuntimeError(
                'self not on top of stack when attempting to exit')

    def preprocessor(self, labels, weights=None):
        """Get a `more_keras.meta_models.preprocessor.Preprocessor`."""
        if isinstance(labels, tf.Tensor):
            labels = (labels,)
        if isinstance(weights, tf.Tensor):
            weights = (weights,)
        return Preprocessor.from_io(
            tuple(self._prebatch_inputs),
            tuple(self._prebatch_outputs),
            tuple(self._batched_inputs),
            tuple(self._batched_feature_outputs),
            labels,
            weights,
        )

    def model(self, outputs):
        """Get the learned model with the given outputs."""
        meta_utils.assert_all_tensors(outputs)
        return tf.keras.models.Model(inputs=tuple(self._model_inputs),
                                     outputs=tuple(outputs))

    def _assert_not_marked(self, tensor, mark_set, target_set):
        if tensor in self._marks[mark_set]:
            raise ValueError(
                'tensor %s is already marked as %s - cannot be marked as %s '
                'as well' % (str(tensor, mark_set, target_set)))

    def _mark(self, tensor, mark, recursive=True):
        existing = self._marks.get(tensor)
        if existing is None:
            self._marks[tensor] = mark
            if recursive:
                for dep in tensor.op.inputs:
                    self._mark(dep, mark)
                # if isinstance(tensor, tf.Tensor):
                #     deps = tensor.op.inputs
                # elif isinstance(tensor, tf.RaggedTensor):
                #     deps = (tensor.flat_values, *tensor.nested_row_splits)
                # # elif isinstance(tensor, _comp.CompositeTensor):
                # # deps = _comp.replace_composites_with_components(tensor)
                # else:
                #     raise TypeError(
                #         'Expected Tensor or CompositeTensor, got {} of type {}'.
                #         format(tensor, type(tensor)))
                # for dep in deps:
                #     self._mark(dep, mark)
        elif existing != mark:
            raise ValueError(
                'attempted to mark tensor %s as %s, but it is already marked '
                'as %s' %
                (tensor, Marks.to_string(mark), Marks.to_string(existing)))

    def prebatch_input(self, shape, dtype, name=None):
        inp = tf.keras.layers.Input(shape=shape,
                                    dtype=dtype,
                                    batch_size=1,
                                    name=name)
        self._prebatch_inputs.append(inp)
        inp = layer_utils.lambda_call(tf.squeeze, inp, axis=0)
        self._mark(inp, Marks.PREBATCH)
        return inp

    def prebatch_inputs_from(self, dataset):
        """
        Get all inputs associated with the first element of dataset.

        Args:
            dataset: with (features, labels), where features and labels
                are tensor structures (lists, tuples, dicts etc.).

        Returns:
            `tf.keras.layers.Input` associated with each tensor of`features`.
        """
        types = dataset.output_types
        # ensure inputs are added in the correct order
        flat_shapes = tf.nest.flatten(dataset.output_shapes)
        flat_types = tf.nest.flatten(types)
        names = [
            '-'.join([str(pi)
                      for pi in p])
            for p in meta_utils.yield_flat_paths(dataset.output_shapes)
        ]
        inputs = tuple(
            self.prebatch_input(s, t, name=n)
            for s, t, n in zip(flat_shapes, flat_types, names))
        return tf.nest.pack_sequence_as(types, inputs)

    def _batched_fixed_tensor(self, tensor):
        assert (isinstance(tensor, tf.Tensor))
        if tensor in self._batched_inputs_dict:
            return self._batched_inputs_dict[tensor]
        self._mark(tensor, Marks.PREBATCH)
        self._prebatch_outputs.append(tensor)
        batched = tf.keras.layers.Input(shape=tensor.shape, dtype=tensor.dtype)
        self._batched_inputs.append(batched)
        self._mark(batched, Marks.BATCHED)
        self._batched_inputs_dict[tensor] = batched
        return batched

    def _batched_tensor(self, tensor):
        shape = tensor.shape
        if len(shape) > 0 and shape[0] is None:
            return self._batched_tensor_with_ragged_leading_dim(tensor)

        return self._batched_fixed_tensor(tensor)

    def _batched_tensor_with_ragged_leading_dim(self, tensor):
        assert (tensor.shape[0] is None)
        if tensor in self._batched_inputs_dict:
            return self._batched_inputs_dict[tensor]
        size = tf.keras.layers.Lambda(
            lambda x: tf.shape(x, out_type=tf.int64)[0])(tensor)
        values = self._batched_fixed_tensor(tensor)
        lengths = self._batched_fixed_tensor(size)
        out = ragged_layers.ragged_from_tensor(values, lengths)
        self._batched_inputs_dict[tensor] = out
        return out

    def _batched_ragged(self, rt):
        if rt in self._batched_inputs_dict:
            return self._batched_inputs_dict[rt]
        size = self._batched_fixed_tensor(
            ragged_layers.ragged_lambda(lambda x: x.nrows())(rt))
        nested_row_lengths = ragged_layers.nested_row_lengths(rt)
        nested_row_lengths = [
            self._batched_tensor(rl) for rl in nested_row_lengths
        ]
        nested_row_lengths = [
            layer_utils.flatten_leading_dims(rl, 2) for rl in nested_row_lengths
        ]
        values = layer_utils.flatten_leading_dims(
            self._batched_tensor(rt.flat_values), 2)
        out = ragged_layers.ragged_from_nested_row_lengths(
            values, [size] + nested_row_lengths)
        self._batched_inputs_dict[rt] = out
        return out

    def _batched(self, rt):
        # handle possible raggedness of possibly ragged tensor
        if isinstance(rt, tf.RaggedTensor):
            return self._batched_ragged(rt)

        assert (isinstance(rt, tf.Tensor))
        return self._batched_tensor(rt)

    def batched(self, tensor):
        """
        Create a structure representing the batched form of the input.

        Example usage:
        ```python
        input_a = b.prebatch_input(shape=(), dtype=tf.float32)
        input_b = b.prebatch_input(shape=(3,), dtype=tf.float32)
        x = 2*a + b  # or the `tf.keras.layers.Lambda` wrapped version in 1.x
        y = x + 1
        batched_x, batched_y = b.batched((x, y))
        print(x.shape)          # (3,)
        print(y.shape)          # (3,)
        print(batched_x.shape)  # (None, 3)
        print(batched_y.shape)  # (None, 3)
        ```

        Args:
            tensor: structure of possibly ragged tensors.

        Returns:
            batched tensor for each element of tensor.
        """
        return tf.nest.map_structure(self._batched, tensor)

    def _as_model_input(self, tensor):
        assert (isinstance(tensor, tf.Tensor))
        if tensor in self._model_inputs_dict:
            return self._model_inputs_dict[tensor]
        self._mark(tensor, Marks.BATCHED)
        self._batched_feature_outputs.append(tensor)
        inp = tf.keras.layers.Input(shape=tensor.shape[1:], dtype=tensor.dtype)
        self._model_inputs.append(inp)
        self._mark(inp, Marks.MODEL)
        self._model_inputs_dict[tensor] = inp
        return inp

    def _as_model_input_single(self, tensor):
        """Marks the tensor as the input of the learned model."""
        if isinstance(tensor, tf.Tensor):
            return self._as_model_input(tensor)
        elif isinstance(tensor, tf.RaggedTensor):
            values = self._as_model_input_single(tensor.flat_values)
            nested_row_splits = [
                self._as_model_input_single(rs)
                for rs in tensor.nested_row_splits
            ]
            return tf.RaggedTensor.from_nested_row_splits(
                values, nested_row_splits)
        else:
            raise ValueError('Unrecognized type "%s"' % tensor)

    def as_model_input(self, tensor):
        return tf.nest.map_structure(self._as_model_input_single, tensor)

    # def _batch_ragged(self, rt):
    #     assert (isinstance(rt, tf.RaggedTensor))
    #     shape = rt.shape
    #     inp_shape = list(shape)
    #     for i in range(rt.ragged_rank):
    #         inp_shape[i] = None
    #     out = tf.keras.layers.Input(shape=inp_shape,
    #                                 dtype=rt.dtype,
    #                                 ragged=True)
    #     return out

    # def _batched(self, tensor):
    #     if tensor in self._batched_inputs_dict:
    #         return self._batched_inputs_dict[tensor]
    #     assert isinstance(tensor, (tf.Tensor, tf.RaggedTensor, tf.SparseTensor))
    #     self._mark(tensor, Marks.PREBATCH)
    #     self._prebatch_outputs.append(tensor)
    #     if isinstance(tensor, tf.RaggedTensor):
    #         batched = self._batch_ragged(tensor)
    #     elif isinstance(tensor, tf.SparseTensor):
    #         raise NotImplementedError
    #     elif isinstance(tensor, tf.Tensor):
    #         batched = tf.keras.layers.Input(shape=tensor.shape,
    #                                         dtype=tensor.dtype)
    #     else:
    #         raise TypeError(
    #             'Only `Tensor`s and `RaggedTensor`s currently supported, got {}'
    #             .format(tensor))

    #     self._batched_inputs.append(batched)
    #     self._mark(batched, Marks.BATCHED)
    #     self._batched_inputs_dict[tensor] = batched
    #     return batched

    # def _as_model_input(self, tensor):
    #     assert (isinstance(tensor,
    #                        (tf.Tensor, tf.RaggedTensor, tf.SparseTensor)))
    #     if tensor.shape.ndims == 0:
    #         raise ValueError('Cannot add tensor with rank 0 as model input')
    #     if tensor in self._model_inputs_dict:
    #         return self._model_inputs_dict[tensor]
    #     self._mark(tensor, Marks.BATCHED)
    #     self._batched_feature_outputs.append(tensor)
    #     inp = tf.keras.layers.Input(shape=tensor.shape[1:],
    #                                 dtype=tensor.dtype,
    #                                 ragged=isinstance(tensor, tf.RaggedTensor),
    #                                 sparse=isinstance(tensor, tf.SparseTensor))
    #     self._model_inputs.append(inp)
    #     self._mark(inp, Marks.MODEL)
    #     self._model_inputs_dict[tensor] = inp
    #     return inp

    # def as_model_input(self, tensor):
    #     """Marks the tensor as the input of the learned model."""
    #     return tf.nest.map_structure(self._as_model_input, tensor)

    def as_batched_model_input(self, tensor):
        """Wrapper around `self.batched` -> `self.as_model_input`."""
        return self.as_model_input(self.batched(tensor))


current = MetaNetworkBuilder.current


def prebatch_input(shape, dtype):
    """See `MetaNetworkBuilder.prebatch_input`."""
    return current().prebatch_input(shape=shape, dtype=dtype)


def prebatch_inputs_from(dataset):
    """See `MetaNetworkBuilder.prebatch_inputs_from`."""
    return current().prebatch_inputs_from(dataset)


def batched(tensor):
    """See `MetaNetworkBuilder.batched`."""
    return current().batched(tensor)


def as_model_input(tensor):
    """See `MetaNetworkBuilder.as_model_input`."""
    return current().as_model_input(tensor)


def as_batched_model_input(tensor):
    """See `MetaNetworkBuilder.as_batched_model_input`."""
    return current().as_batched_model_input(tensor)


def preprocessor(labels):
    """See `MetaNetworkBuilder.preprocessor`."""
    return current().preprocessor(labels)


def model(outputs):
    """See `MetaNetworkBuilder.model`."""
    return current().model(outputs)


def get_mark(tensor):
    """See `MetaNetworkBuilder.get_mark`."""
    return current().get_mark(tensor)
