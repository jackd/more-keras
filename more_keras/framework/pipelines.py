from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gin
import tensorflow as tf
from more_keras.framework.problems import core
from more_keras.framework import functions
from more_keras.utils import identity

AUTOTUNE = tf.data.experimental.AUTOTUNE
NO_REPEAT = 'NO_REPEAT'


@gin.configurable(module='mk.framework')
class Pipeline(object):

    def __init__(self,
                 batch_size=None,
                 repeats=NO_REPEAT,
                 shuffle_buffer=None,
                 drop_remainder=False,
                 map_fn=None,
                 prefetch_buffer=AUTOTUNE,
                 num_parallel_calls=AUTOTUNE,
                 output_spec_fn=identity):
        self.batch_size = batch_size
        self.repeats = repeats
        self.shuffle_buffer = shuffle_buffer
        self.drop_remainder = drop_remainder
        self.prefetch_buffer = prefetch_buffer
        self.num_parallel_calls = num_parallel_calls
        self.output_spec_fn = functions.get(output_spec_fn)
        self.map_fn = functions.get(map_fn)

    def output_spec(self, input_spec):
        if self.output_spec_fn is None:
            return input_spec
        else:
            return self.output_spec_fn(input_spec)

    def preprocess_dataset(self, dataset):
        if self.repeats != NO_REPEAT:
            dataset = dataset.repeat()
        if self.shuffle_buffer is not None:
            dataset = dataset.shuffle(self.shuffle_buffer)
        if self.map_fn is not None:
            dataset = dataset.map(self.map_fn, self.num_parallel_calls)
        if self.batch_size is not None:
            dataset = dataset.batch(self.batch_size,
                                    drop_remainder=self.drop_remainder)
        if self.prefetch_buffer:
            dataset = dataset.prefetch(self.prefetch_buffer)
        return dataset

    def __call__(self, dataset):
        return self.preprocess_dataset(dataset)

    def get_generator(self, dataset_fn):
        import tensorflow_datasets as tfds
        graph = tf.Graph()
        with graph.as_default():  # pylint: disable=not-context-manager
            dataset = self.preprocess_dataset(dataset_fn())
        return tfds.as_numpy(dataset, graph=graph)

    def get_config(self):
        return dict(batch_size=self.batch_size,
                    repeats=self.repeats,
                    shuffle_buffer=self.shuffle_buffer,
                    map_fn=functions.serialize(self.map_fn),
                    prefetch_buffer=self.prefetch_buffer,
                    num_parallel_calls=self.num_parallel_calls,
                    output_spec_fn=functions.serialize(self.output_spec_fn),
                    drop_remainder=self.drop_remainder)

    @classmethod
    def from_config(cls, config):
        return cls(**config)
