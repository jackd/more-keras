from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def ious(confusion):
    intersection = np.diag(confusion)
    union = (
        np.sum(confusion, axis=0) + np.sum(confusion, axis=1) - intersection)
    return intersection / union
