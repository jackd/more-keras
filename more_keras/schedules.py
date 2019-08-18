from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class ExponentialDecayTowards(tf.keras.optimizers.schedules.ExponentialDecay):
    """Exponential decay scheduler with lower bound."""

    def __init__(self,
                 initial_learning_rate,
                 decay_steps,
                 decay_rate,
                 asymptote=0,
                 clip_value=None,
                 staircase=False,
                 name=None):
        super(ExponentialDecayTowards, self).__init__(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name=name,
        )
        self.asymptote = asymptote
        self.clip_value = clip_value

    def __call__(self, step):
        return exponential_decay_towards(
            step,
            initial_value=self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            asymptote=self.asymptote,
            clip_value=self.clip_value,
            staircase=self.staircase,
            impl=tf,
        )

    def get_config(self):
        config = super(ExponentialDecayTowards, self).get_config()
        config['asyptote'] = self.asymptote
        config['clip_value'] = self.clip_value
        return config


def exponential_decay(step,
                      initial_value,
                      decay_steps,
                      decay_rate,
                      min_value=None,
                      staircase=False,
                      impl=tf):
    """
    Exponential decay schedule.

    Args:
        step: primary input
        initial_value: value when step = 0
        decay_steps: number of steps for each full decay
        decay_rate: rate of decay per `decay_steps`
        min_value: minimum value (or None for no clipping)
        staircase: if True, the floor of steps / decay_steps is used
        impl: anything with a 'floor' and 'maximum' function, e.g. np or tf

    Returns:
        possibly clipped exponentially decayed value.
    """
    exponent = step / decay_steps
    if staircase:
        exponent = impl.floor(exponent)
    value = initial_value * decay_rate**exponent
    if min_value is not None:
        value = impl.maximum(value, min_value)
    return value


def exponential_decay_towards(step,
                              initial_value,
                              decay_steps,
                              decay_rate,
                              asymptote=1.0,
                              clip_value=None,
                              staircase=False,
                              impl=tf):
    """
    Return exponential approaching the given asymptote.

    See expoential_decay.
    """
    kwargs = dict(decay_steps=decay_steps,
                  decay_rate=decay_rate,
                  staircase=staircase,
                  impl=impl)
    if asymptote > initial_value:
        return asymptote - exponential_decay(
            step,
            asymptote - initial_value,
            min_value=None if clip_value is None else asymptote - clip_value,
            **kwargs)
    else:
        return asymptote + exponential_decay(
            step,
            initial_value - asymptote,
            min_value=None if clip_value is None else clip_value - asymptote,
            **kwargs)
