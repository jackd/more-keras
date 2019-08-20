from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import gin
import six
from more_keras.tf_compat import is_v1
from more_keras.callbacks.model_checkpoint import LATEST


@gin.configurable
class SaverCallback(tf.keras.callbacks.Callback):
    """Callback using a standard `tf.train.Saver`."""

    def __init__(self,
                 directory,
                 save_freq='epoch',
                 max_to_keep=5,
                 load_weights_on_restart=False,
                 **saver_kwargs):
        if not is_v1:
            raise RuntimeError(
                'SaverCallback is only usable in tensorflow v1 - see '
                'CheckpointManagerCallback for substitute')
        if save_freq == 'epoch':
            save_freq = 1
        elif not isinstance(save_freq, int):
            raise ValueError(
                'save_freq must be an int or "epoch", got {}'.format(save_freq))
        self._directory = directory
        self.load_weights_on_restart = load_weights_on_restart
        self._save_freq = save_freq
        self._last_save = None
        self._epoch = None
        self._chkpt_format = os.path.join(self._directory, 'model-{epoch:05d}')
        self._started = False
        self._saver = None
        self._max_to_keep = max_to_keep
        self._saver_kwargs = saver_kwargs

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=self._max_to_keep,
                                         **self._saver_kwargs)
        return self._saver

    def reset_state(self):
        self._started = False

    def on_train_begin(self, logs=None):
        self._started = True
        if self.load_weights_on_restart:
            self.restore()

    def on_test_begin(self, logs=None):
        if not self._started and self.load_weights_on_restart:
            self.restore()

    def on_predict_begin(self, logs=None):
        if not self._started and self.load_weights_on_restart:
            self.restore()

    @property
    def directory(self):
        return self._directory

    def checkpoint(self, checkpoint_or_epoch_or_latest=LATEST):
        if checkpoint_or_epoch_or_latest is None:
            return None
        elif checkpoint_or_epoch_or_latest == LATEST:
            return tf.train.latest_checkpoint(self._directory)
        elif isinstance(checkpoint_or_epoch_or_latest, int):
            return self._chkpt_format.format(
                epoch=checkpoint_or_epoch_or_latest)
        elif isinstance(checkpoint_or_epoch_or_latest, six.string_types):
            if tf.io.gfile.exists(
                    '{}.index'.format(checkpoint_or_epoch_or_latest)):
                return checkpoint_or_epoch_or_latest
            else:
                raise ValueError(
                    'No files found for checkpoint "{}". Files in folder: {}'.
                    format(checkpoint_or_epoch_or_latest,
                           os.listdir(self._directory)))
        else:
            raise ValueError(
                'Unrecognized value for checkpoint_or_epoch_or_latest, {}'.
                format(checkpoint_or_epoch_or_latest))

    def epoch(self, checkpoint=LATEST):
        checkpoint = self.checkpoint(checkpoint)
        if checkpoint is None:
            return checkpoint

        _, filename = os.path.split(checkpoint)
        epoch = int(filename[6:11])
        return epoch

    def restore(self, checkpoint=LATEST):
        checkpoint = self.checkpoint(checkpoint)
        if checkpoint is None:
            return
        self.saver.restore(tf.keras.backend.get_session(), checkpoint)

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1
        self._epoch = epoch
        if epoch % self._save_freq == 0 and self._last_save != epoch:
            self._save()

    def on_train_end(self, logs=None):
        self._save()

    def _save(self):
        if self._epoch is not None and (self._last_save is None or
                                        self._last_save < self._epoch):
            self.saver.save(tf.keras.backend.get_session(),
                            self._chkpt_format.format(epoch=self._epoch))
            self._last_save = self._epoch
