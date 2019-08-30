from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import dill
import functools
import importlib
import six
import tensorflow as tf


def _serialize_func(func):
    name = func.__name__
    if name == '<lambda>':
        raise ValueError('Cannot serialize lambda expression')
    if not dill.pickles(func):
        raise ValueError('Cannot pickle func {}'.format(func))
    module = func.__module__
    return dict(class_name=name, module=module)


def _update_without_overwrite(target, extras):
    for k, v in extras.items():
        if k in extras:
            raise ValueError('Cannot add key {} - already present'.format(k))
        target[k] = v
    return target


ATOMIC_TYPES = (int, float, bool) + six.string_types


def is_serializable(obj):
    if isinstance(obj, ATOMIC_TYPES):
        return True
    if isinstance(obj, dict):
        return all(
            is_serializable(k) and is_serializable(v) for k, v in obj.items())
    if isinstance(obj, list):
        return all(is_serializable(v) for v in obj)
    else:
        return False


def serialize(func):
    if func is None:
        return None

    if not callable(func):
        raise ValueError('Cannot serialize non-callable {}'.format(func))

    if isinstance(func, functools.partial):
        keywords = func.func
        for k, v in keywords.items():
            if not is_serializable(v):
                raise ValueError('Cannot serialize {} for keyword {}'.format(
                    v, k))
        config = serialize(func.func)
        if 'keywords' in config:
            _update_without_overwrite(config['keywords'], keywords)
        else:
            config['keywords'] = keywords
    else:
        config = _serialize_func(func)
    return config


def deserialize(identifier):
    if isinstance(identifier, six.string_types):
        return tf.keras.utils.deserialize_keras_object(identifier)
    elif isinstance(identifier, dict):
        name = identifier['class_name']
        module = identifier['module']
        func = getattr(importlib.import_module(module), name)
        if 'keywords' in identifier:
            keywords = identifier['keywords']
            return functools.partial(func, **keywords)
        else:
            return func
    else:
        raise TypeError('Can only deserialize dicts or strings')


def get(identifier):
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, collections.Mapping + six.string_types):
        return deserialize(identifier)
    else:
        raise TypeError('Cannot get function {}'.format(identifier))
