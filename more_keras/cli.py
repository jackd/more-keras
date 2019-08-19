"""DEFAULT initial values are replaced with relevant FLAG values."""
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

DEFAULT = '__default__'

flags.DEFINE_string(
    'config_dir',
    None,
    help='Root config directory. `config` paths are relative to this.')
flags.DEFINE_multi_string(
    'config', [],
    help=('List of paths to the config files. Relative paths are relative to '
          '`config_dir` if provided.'))
flags.DEFINE_multi_string(
    'mk_config', [],
    help='List of paths to the config files relative to mk config dir.')
flags.DEFINE_multi_string('bindings', [],
                          'Newline separated list of gin parameter bindings.')


def parse_cli_config(finalize_config=True):
    """
    Parse config from flags.

    Parses mk_configs, then configs, then bindings.
    """
    from more_keras import config
    FLAGS = flags.FLAGS
    config_files = config.fix_configs(FLAGS.config)
    mk_config_files = config.fix_configs(FLAGS.mk_config)
    config.parse_mk_config(mk_config_files)
    config.parse_relative_config(config_dir=FLAGS.config_dir,
                                 config_files=config_files)
    gin.parse_config(FLAGS.bindings)
    if finalize_config:
        gin.finalize()


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
