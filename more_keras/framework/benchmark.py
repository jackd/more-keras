from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import gin
from absl import logging
import timeit


@gin.configurable(module='mk.framework')
def benchmark(op_fn,
              burn_iters=2,
              min_iters=10,
              store_trace=False,
              store_memory_usage=True,
              name=None,
              extras=None,
              mbs=0):
    with tf.Graph().as_default():
        ops = op_fn()
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            bm = tf.test.Benchmark()
            bm.run_op_benchmark(sess,
                                ops,
                                burn_iters=burn_iters,
                                min_iters=min_iters,
                                store_trace=store_trace,
                                store_memory_usage=store_memory_usage,
                                name=name,
                                extras=extras,
                                mbs=mbs)
    return bm
