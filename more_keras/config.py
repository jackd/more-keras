from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import logging
import contextlib
from more_keras.null_context import null_context
import gin
import six

MK_CONFIG_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), 'configs'))
os.environ['MK_CONFIG_DIR'] = MK_CONFIG_DIR


@contextlib.contextmanager
def change_dir_context(path):
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


def fix_configs(config_files):
    """
    Convert a string/list of strings into a list of strings in canonical form.

    In order:
        - converts a single string to a single-element list
        - concatenates the result of splitting over new lines
        - expands variables
        - expands user
        - filters out empty strings
    """
    import numpy as np
    if config_files == []:
        return config_files
    if isinstance(config_files, six.string_types):
        config_files = [config_files]
    config_files = np.concatenate([c.split('\n') for c in config_files])
    config_files = (os.path.expanduser(
        os.path.expandvars(p if p.endswith('.gin') else '{}.gin'.format(p)))
                    for p in config_files)
    return [p for p in config_files if p != '']


def parse_relative_config(config_dir, config_files):
    """Parse config files relative to root config_dir."""
    config_files = fix_configs(config_files)
    context = null_context() if config_dir is None else change_dir_context(
        config_dir)
    with context:
        for f in config_files:
            gin.parse_config_file(f)


def parse_mk_config(config_files):
    parse_relative_config(MK_CONFIG_DIR, config_files)
