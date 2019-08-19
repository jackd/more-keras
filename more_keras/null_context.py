from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib


@contextlib.contextmanager
def null_context():
    yield
