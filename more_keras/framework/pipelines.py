from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gin
import functools
import tensorflow as tf
from more_keras.framework.problems import core
from more_keras.framework import functions
from more_keras import spec
import collections

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
                 post_batch_map_fn=None,
                 prefetch_buffer=AUTOTUNE,
                 num_parallel_calls=AUTOTUNE):
        self.batch_size = batch_size
        self.repeats = repeats
        self.shuffle_buffer = shuffle_buffer
        self.drop_remainder = drop_remainder
        self.prefetch_buffer = prefetch_buffer
        self.num_parallel_calls = num_parallel_calls
        self.map_fn = functions.get(map_fn)
        self.post_batch_map_fn = functions.get(post_batch_map_fn)

    def rebuild(self, **kwargs):
        return Pipeline(
            batch_size=kwargs.get('batch_size', self.batch_size),
            repeats=kwargs.get('repeats', self.repeats),
            shuffle_buffer=kwargs.get('shuffle_buffer', self.shuffle_buffer),
            drop_remainder=kwargs.get('drop_remainder', self.drop_remainder),
            map_fn=kwargs.get('map_fn', self.map_fn),
            post_batch_map_fn=kwargs.get('post_batch_map_fn',
                                         self.post_batch_map_fn),
            prefetch_buffer=kwargs.get('prefetch_buffer', self.prefetch_buffer),
            num_parallel_calls=kwargs.get('num_parallel_calls',
                                          self.num_parallel_calls),
        )

    def element_spec(self, input_spec):
        with tf.Graph().as_default():

            def f():
                raise NotImplementedError()

            types = tf.nest.map_structure(lambda x: x.dtype, input_spec)
            shapes = tf.nest.map_structure(lambda x: x.shape, input_spec)
            dataset = tf.data.Dataset.from_generator(f, types, shapes)
            dataset = self.map_and_batch(dataset)
            element_spec = spec.element_spec(dataset)
        return element_spec

    def map_and_batch(self, dataset):
        if self.map_fn is not None:
            dataset = dataset.map(self.map_fn, self.num_parallel_calls)
        if self.batch_size is not None:
            dataset = dataset.batch(self.batch_size,
                                    drop_remainder=self.drop_remainder)
        if self.post_batch_map_fn is not None:
            dataset = dataset.map(self.post_batch_map_fn,
                                  self.num_parallel_calls)
        return dataset

    def preprocess_dataset(self, dataset):
        if self.shuffle_buffer is not None:
            dataset = dataset.shuffle(self.shuffle_buffer)
        if self.repeats != NO_REPEAT:
            dataset = dataset.repeat(self.repeats)
        dataset = self.map_and_batch(dataset)
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
                    post_batch_map_fn=functions.serialize(
                        self.post_batch_map_fn),
                    prefetch_buffer=self.prefetch_buffer,
                    num_parallel_calls=self.num_parallel_calls,
                    drop_remainder=self.drop_remainder)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def map(self, map_fn, post_batch_map_fn=None):
        """
        Get a new pipeline with intercepting map functions.

        Order of map_and_batch of new pipeline is
            self.map_fn
            map_fn
            batch
            post_batch_map_fn
            self.post_batch_map_fn

        Args:
            map_fn: map function applied between self.map_fn and batching
            post_batch_map_fn: map function applied between batching and
                self.post_batch_map_fn

        Returns:
            new Pipeline instance.
        """

        def maybe_chain(*fns):
            fns = tuple(fn for fn in fns)
            if len(fns) == 0:
                return None
            elif len(fns) == 1:
                return fns[0]
            else:
                return functools.partial(chain_map_fns, fns=fns)

        return self.rebuild(map_fn=maybe_chain(self.map_fn, map_fn),
                            post_batch_map_fn=maybe_chain(
                                post_batch_map_fn, self.post_batch_map_fn))


def chain_map_fns(*args, fns=()):
    """
    Apply fns in order to `features, labels`.

    Each function is expected to return a tuple of (features, labels).
    """
    for fn in fns:
        if fn is not None:
            args = fn(*args)
    return args


@gin.configurable(module='mk.framework')
def map_pipeline(pipeline, map_fn, post_batch_map_fn=None):
    return pipeline.map(map_fn, post_batch_map_fn)


@gin.configurable(module='mk.framework')
def pipeline_benchmark(problem, split, pipeline):
    if isinstance(pipeline, dict):
        pipeline = pipeline[split]
    with problem:
        dataset = pipeline.preprocess_dataset(problem.get_base_dataset(split))
        return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
