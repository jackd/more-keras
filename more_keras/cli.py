from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl import logging
import contextlib
import six
import numpy as np
import gin

flags.DEFINE_multi_string('config_files', [],
                          'config files appended to positional args.')
flags.DEFINE_multi_string('bindings', [],
                          'Newline separated list of gin parameter bindings.')
flags.DEFINE_boolean('incl_rel',
                     default=True,
                     help='Whether or not to enable_relative_includes')
flags.DEFINE_boolean('expand_vars',
                     default=True,
                     help='Whether or not to enable vars/user in includes')


@gin.configurable(module='mk')
def logging_config(to_file=True, log_dir=None, program_name='more_keras'):
    if to_file:
        log_dir = os.path.expanduser(os.path.expandvars(log_dir))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.info('Logging to {}'.format(log_dir))
        logging.get_absl_handler().use_absl_log_file(log_dir=log_dir,
                                                     program_name=program_name)


def get_gin_summary(argv):
    """
    Collect GinSummary from command line args

    Args:
        argv: clargs that weren't passed by absl. Interpretted as config files
        finalize_config: if True, config is finalized after parsing.

    Returns:
        GinSummary object
    """
    from more_keras.gin_utils.summary import GinSummary
    FLAGS = flags.FLAGS
    return GinSummary(os.getcwd(), FLAGS.incl_rel, FLAGS.expand_vars,
                      argv[1:] + FLAGS.config_files, FLAGS.bindings)


def assert_clargs_parsed(argv, max_allowed=1):
    if len(argv) > max_allowed:
        raise ValueError(
            'Unpassed command line args exceeded limit of {}: {}'.format(
                max_allowed, ' '.join(argv)))


@gin.configurable(module='mk')
def main(fn=None):
    if fn is None:
        logging.error('`main.fn` is not configured.')
    fn()
