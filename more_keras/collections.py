from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import MutableSequence


class DelegatingMutableSequence(MutableSequence):

    def __init__(self, values=[]):
        self._values = values

    def __getitem__(self, index):
        return self._values[index]

    def __setitem__(self, index, value):
        self._values[index] = value

    def insert(self, index, value):
        self._values.insert(index, value)

    def __delitem__(self, index):
        del self._values[index]

    def __len__(self):
        return len(self._values)


class ValidatingSequence(DelegatingMutableSequence):

    def __init__(self, validator, values=None):
        if not callable(validator):
            raise ValueError(
                'validator must be callable, got {}'.format(validator))
        self._validator = validator
        if values is None:
            values = []
        else:
            values = list(values)
            for v in values:
                self._validate(v)
        super(ValidatingSequence, self).__init__(values)

    def _validate(self, value):
        if not self._validator(value):
            raise ValueError('Value {} failed validation'.format(value))
        return value

    def __setitem__(self, index, value):
        super(ValidatingSequence, self).__setitem__(index,
                                                    self._validate(value))

    def insert(self, index, value):
        self._values.insert(index, self._validate(value))


def isinstance_validator(types):

    def f(x):
        if not isinstance(x, types):
            raise ValueError('Expected value of type {}, got {}'.format(
                types, x))
        return True

    return f


def typed_sequence(types, initial_values=None):
    return ValidatingSequence(isinstance_validator(types), initial_values)
