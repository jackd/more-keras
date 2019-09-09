from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import copy
import gin
import tensorflow as tf
import tensorflow_datasets as tfds
import six
from more_keras import spec


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
                 element_spec=None,
                 output_spec=None):
        if isinstance(loss, (list, tuple)):
            self.loss = [tf.keras.losses.get(l) for l in loss]
        else:
            self.loss = tf.keras.losses.get(loss)
        if metrics is None or len(metrics) == 0:
            self.metrics = []
        elif isinstance(metrics[0], (list, tuple)):
            self.metrics = [
                [tf.keras.metrics.get(mi) for mi in m] for m in metrics
            ]
            if objective is None:
                objective = 'val_{}'.format(self.metrics[0][0].name)
        else:
            self.metrics = [tf.keras.metrics.get(m) for m in metrics]
            if objective is None:
                objective = 'val_{}'.format(self.metrics[0].name)
        self.objective = Objective.get(objective)
        self._element_spec = element_spec
        self._output_spec = output_spec

    @property
    def element_spec(self):
        if self._element_spec is None:
            with tf.Graph().as_default():
                self._element_spec = spec.element_spec(
                    self.get_base_dataset('train'))
        return copy.deepcopy(self._element_spec)

    @property
    def output_spec(self):
        if self._output_spec is None:
            self._output_spec = self.element_spec[1]  # default: same as labels
        return self._output_spec

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
            objective=None if objective is None else objective.get_config())

    # def post_batch_map(self, labels, weights=None):
    #     if weights is None:
    #         return labels
    #     else:
    #         return labels, weights
