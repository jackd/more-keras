from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import logging
import contextlib
import gin
import six

MK_CONFIG_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', 'configs'))
if 'MK_CONFIG' in os.environ:
    logging.warning(
        'MK_CONFIG environment variable defined. '
        '`more_keras.gin_utils.parse_mk_config` may act surprisingly')
else:
    os.environ['MK_CONFIG'] = MK_CONFIG_DIR


@contextlib.contextmanager
def change_dir_context(path):
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


def fix_paths(config_files):
    """
    Convert a string/list of strings into a list of strings in canonical form.

    In order:
        - converts a single string to a single-element list
        - concatenates the result of splitting over new lines and comma
        - filters out empty strings
    """
    import numpy as np
    if config_files == []:
        return config_files
    if isinstance(config_files, six.string_types):
        config_files = [config_files]
    config_files = np.concatenate([c.split('\n') for c in config_files])
    config_files = np.concatenate([c.split(',') for c in config_files])
    config_files = (c.strip() for c in config_files)
    config_files = (c for c in config_files if c != '')
    config_files = (  # add missing .gin extension
        p if p.endswith('.gin') else '{}.gin'.format(p) for p in config_files)
    return [p for p in config_files if p.strip() != '']


def fix_bindings(bindings):
    """Convert a string/list of strings into a single string of bindings."""
    if isinstance(bindings, (list, tuple)):
        bindings = '\n'.join(bindings)
    return bindings
