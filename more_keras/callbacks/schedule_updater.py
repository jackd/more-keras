from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from more_keras.collections import typed_sequence
from more_keras.utils import UpdateFrequency
K = tf.keras.backend


class ScheduleUpdater(tf.keras.callbacks.Callback):
    """
    Callback for updating variable(s) according to a schedule after each epoch.

    Note you can mutate variables after creation, so long as only `tf.Variable`s
    are added.
    """

    def __init__(self,
                 schedule,
                 variables=None,
                 variables_func=None,
                 update_freq=UpdateFrequency.EPOCH,
                 logs_key=None):
        """
        Args:
            schedule: function mapping step to variable values. Step is either
                the epoch count or the optimizer's iteration count, depending
                on update_freq.
            variables: iterable of `tf.Variable`s of None.
            variables_func: function mapping model to an iterable of variables,
                or `None`. These variables are updated as well as the base
                variables, though the set of variables affected is reset on
                each model set.
        """
        self._variables = typed_sequence(tf.Variable, variables)
        self._model_variables = None
        if not callable(schedule):
            schedule = tf.keras.utils.deserialize_keras_object(schedule)
        self.schedule = schedule
        self._variables_func = variables_func
        self.update_freq = update_freq
        self.logs_key = logs_key

    def set_model(self, model):
        super(ScheduleUpdater, self).set_model(model)
        if self._variables_func is not None:
            self._model_variables = self._variables_func(model)

    @property
    def update_freq(self):
        return self._update_freq

    @update_freq.setter
    def update_freq(self, update_freq):
        UpdateFrequency.validate(update_freq)
        self._update_freq = update_freq

    @property
    def variables(self):
        return self._variables

    def update(self, step):
        value = self.schedule(step)
        if isinstance(value, tf.Tensor):
            raise ValueError('schedule should be a normal numpy function')
        for variable in self.variables:
            K.set_value(variable, value)
        for variable in self._model_variables:
            K.set_value(variable, value)
        self.value = value

    def on_epoch_begin(self, epoch, logs=None):
        if self.update_freq == UpdateFrequency.EPOCH:
            self.update(epoch)
        return self._update_logs(logs)

    def on_epoch_end(self, epoch, logs=None):
        return self._update_logs(logs)

    def _update_logs(self, logs):
        key = self.logs_key
        if key is not None:
            if logs is None:
                logs = {}
            if key in logs:
                raise KeyError(
                    'Attempted to overwrite key {} in logs'.format(key))
            logs[key] = self.value
        return logs

    def on_train_batch_begin(self, batch, logs=None):
        if self.update_freq == UpdateFrequency.BATCH:
            self.update(K.get_value(self.model.optimizer.iterations))
        return self._update_logs(logs)
