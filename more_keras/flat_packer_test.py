from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import test_util  # pylint: disable=import-error
from more_keras.test_utils import RaggedTestCase
from more_keras.flat_packer import FlatPacker
from more_keras.spec import to_spec


@test_util.run_all_in_graph_and_eager_modes
class FlatPackerTest(RaggedTestCase):
    # class FlatPackerTest(object):

    def _test_pack_unpack(self, data, spec=None):
        packer = FlatPacker(spec or tf.nest.map_structure(to_spec, data))
        packed = packer.pack(data)
        unpacked = packer.unpack(packed)
        self.assertEqual(packed.shape.ndims, 1)
        tf.nest.map_structure(self.assertRaggedEqual, data, unpacked)

    def test_simple(self):
        spec = dict(
            features=tf.TensorSpec(shape=(3, 4), dtype=tf.float32),
            labels=tf.TensorSpec(shape=(None, 3), dtype=tf.int64),
        )
        packer = FlatPacker(spec)
        data = dict(features=tf.constant(np.random.uniform(size=((3, 4))),
                                         dtype=tf.float32),
                    labels=tf.constant(np.random.uniform(size=(5, 3)),
                                       dtype=tf.int64))

        packed = packer.pack(data)
        unpacked = packer.unpack(packed)
        data, unpacked = self.evaluate((data, unpacked))
        self.assertAllEqual(data['features'], unpacked['features'])
        self.assertAllEqual(data['labels'], unpacked['labels'])

    def test_nested(self):
        data = dict(features=dict(
            x=tf.constant(np.random.uniform(size=((3, 4))), dtype=tf.float32),
            y=tf.constant(np.random.uniform(size=((2,))), dtype=tf.float32),
        ),
                    labels=tf.constant(np.random.uniform(size=(5, 3)),
                                       dtype=tf.int64))
        self._test_pack_unpack(data)

    def test_ragged(self):
        data = tf.RaggedTensor.from_row_splits([2, 3, 4], [0, 2, 3])
        self._test_pack_unpack(data)

    def test_multi_ragged(self):
        data = tf.RaggedTensor.from_nested_row_splits([2, 3, 4, 5],
                                                      [[0, 2, 3], [0, 2, 3, 4]])
        self._test_pack_unpack(data)


if __name__ == '__main__':
    tf.test.main()
