from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow_datasets as tfds
from more_keras.framework.problems.core import Problem
import six


class TfdsProblem(Problem):

    def __init__(self,
                 builder,
                 loss,
                 metrics=(),
                 objective=None,
                 input_spec=None,
                 output_spec=None,
                 as_supervised=True,
                 split_map=None):
        if isinstance(builder, six.string_types):
            builder = tfds.builder(builder)
        self.builder = builder

        self.as_supervised = as_supervised
        if input_spec is None or output_spec is None and as_supervised:
            info = self.builder.info
            inp, out = (info.features[k] for k in info.supervised_keys)
            if input_spec is None:
                input_spec = tf.keras.layers.InputSpec(shape=inp.shape,
                                                       dtype=inp.dtype)
            if output_spec is None:
                if hasattr(out, 'num_classes'):
                    # classification
                    shape = out.shape + (out.num_classes,)
                    dtype = tf.float32
                else:
                    shape = out.shape
                    dtype = out.dtype
                output_spec = tf.keras.layers.InputSpec(shape=shape,
                                                        dtype=dtype)
        if split_map is None:
            split_map = {}
        self.split_map = split_map
        super(TfdsProblem, self).__init__(
            loss=loss,
            metrics=metrics,
            objective=objective,
            input_spec=input_spec,
            output_spec=output_spec,
        )

    def get_config(self):
        config = super(TfdsProblem, self).get_config()
        builder = self.builder
        config['builder'] = (builder.name if
                             builder.builder_config is None else '{}/{}'.format(
                                 builder.name, builder.builder_config.name))
        config['split_map'] = self.split_map
        config['as_supervised'] = self.as_supervised
        return config

    def _split(self, split):
        return self.split_map.get(split, split)

    def _examples_per_epoch(self, split):
        split = self._split(split)

        def get(split):
            return self.builder.info.splits[split].num_examples

        if isinstance(split, (tfds.core.splits.NamedSplit,) + six.string_types):
            return get(split)
        else:
            # fractional split?
            # https://github.com/tensorflow/datasets/tree/master/docs/splits.md
            acc = 0
            for k, (start, end) in split.items():
                percent = round((end - start) * 100) / 100
                acc += round(get(k) * percent)
            return acc

    def _get_base_dataset(self, split):
        split = self._split(split)
        if isinstance(split, dict):
            RI = tfds.core.tfrecords_reader.ReadInstruction
            ri = None
            for k, (from_, to) in split.items():
                nex = RI(k, from_=from_ * 100, to=to * 100, unit='%')
                if ri is None:
                    ri = nex
                else:
                    ri = ri + nex
            split = ri

        return self.builder.as_dataset(split=split,
                                       as_supervised=self.as_supervised)
