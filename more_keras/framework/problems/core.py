from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import gin
import tensorflow as tf
import tensorflow_datasets as tfds
import six


@gin.configurable(module='mk.framework')
class Objective(object):

    def __init__(self, name, mode='max'):
        self.name = name
        self.mode = mode

    def get_config(self):
        return dict(name=self.name, mode=self.mode)

    @classmethod
    def get(self, identifier):
        if isinstance(identifier, Objective):
            return identifier
        if isinstance(identifier, (list, tuple)):
            return Objective(*identifier)
        elif isinstance(identifier, dict):
            return Objective(**identifier)
        elif isinstance(identifier, six.string_types):
            return Objective(identifier)
        else:
            raise TypeError(
                'Cannot convert identifier {} into an Objective'.format(
                    identifier))


class Problem(object):

    def __init__(self,
                 loss,
                 metrics=(),
                 objective=None,
                 input_spec=None,
                 output_spec=None,
                 labels_spec=None,
                 weights_spec=None):
        self.loss = tf.keras.losses.get(loss)
        self.metrics = [tf.keras.metrics.get(m) for m in metrics]
        if objective is None and len(self.metrics) > 0:
            objective = 'val_{}'.format(self.metrics[0].name)
        self.objective = Objective.get(objective)
        self.input_spec = get_input_spec(input_spec)
        self.output_spec = get_input_spec(output_spec)
        self.labels_spec = get_input_spec(labels_spec)
        self.weights_spec = get_input_spec(weights_spec)

    def batch_labels_and_weights(self, labels, weights=None):
        from more_keras.meta_models import builder as b
        return tf.nest.map_structure(
            lambda tensor: None
            if tensor is None else b.batched(tensor), (labels, weights))

    @abc.abstractmethod
    def _examples_per_epoch(self, split):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_base_dataset(self, split):
        raise NotImplementedError

    def examples_per_epoch(self, split=tfds.Split.TRAIN, batch_size=None):
        return tf.nest.map_structure(self._examples_per_epoch, split)

    def get_base_dataset(self, split=tfds.Split.TRAIN):
        return tf.nest.map_structure(self._get_base_dataset, split)

    def get_config(self):
        objective = self.objective
        return dict(
            loss=tf.keras.utils.serialize_keras_object(self.loss),
            metrics=[
                tf.keras.utils.serialize_keras_object(m) for m in self.metrics
            ],
            objective=None if objective is None else objective.get_config(),
            input_spec=get_input_spec_config(self.input_spec),
            output_spec=get_input_spec_config(self.output_spec))


def get_input_spec_config(input_spec):
    if input_spec is None:
        return None
    return dict(dtype=repr(input_spec.dtype)[3:],
                shape=input_spec.shape,
                ndim=input_spec.ndim,
                max_ndim=input_spec.max_ndim,
                min_ndim=input_spec.min_ndim,
                axes=input_spec.axes)


def get_input_spec(identifier):
    if identifier is None or isinstance(identifier, tf.keras.layers.InputSpec):
        return identifier
    elif isinstance(identifier, dict):
        if identifier.get('class_name') == 'InputSpec':
            return tf.keras.layers.InputSpec(**identifier['config'])
        else:
            return {k: get_input_spec(v) for k, v in identifier.items()}
    else:
        raise TypeError(
            'Cannot convert value {} to InputSpec'.format(identifier))
