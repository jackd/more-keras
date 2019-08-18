from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
tf.keras.callbacks.TensorBoard
LATEST = 'LATEST'


class BetterModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """
    ModelCheckpoint with slightly extended interface.

    Particularly useful for starting setting the appropriate initial_epoch.

    Also loads weights at beginning of test/prediction unless training was
    started with training.
    """

    def __init__(self,
                 directory,
                 save_freq='epoch',
                 load_weights_on_restart=False,
                 max_to_keep=5,
                 **kwargs):
        directory = os.path.expandvars(os.path.expanduser(directory))
        if not os.path.isdir(directory):
            os.makedirs(directory)
        self._directory = directory
        filepath = os.path.join(directory, 'model-{epoch:05d}.h5')
        self._max_to_keep = max_to_keep
        self._started = False
        super(BetterModelCheckpoint,
              self).__init__(filepath=filepath,
                             save_freq=save_freq,
                             load_weights_on_restart=load_weights_on_restart,
                             **kwargs)

    def on_train_begin(self, logs=None):
        self._started = True
        return super(BetterModelCheckpoint, self).on_train_begin(logs=logs)

    def on_test_begin(self, logs=None):
        if not self._started and self.load_weights_on_restart:
            self.restore()

    def on_predict_begin(self, logs=None):
        if not self._started and self.load_weights_on_restart:
            self.restore()

    @property
    def directory(self):
        return self._directory

    @property
    def latest_checkpoint(self):
        return self._get_most_recently_modified_file_matching_pattern(
            self.filepath)

    @property
    def latest_epoch(self):
        checkpoint = self.latest_checkpoint
        return None if checkpoint is None else self.epoch(checkpoint)

    def epoch(self, checkpoint):
        return int(checkpoint[-8:-3])

    def restore(self, checkpoint=LATEST):
        if isinstance(checkpoint, int):
            checkpoint = self.filepath.format(epoch=checkpoint)
        elif checkpoint == LATEST:
            checkpoint = self.latest_checkpoint
        self.model.load_weights(checkpoint)

    def _save_model(self, epoch, logs):
        super(BetterModelCheckpoint, self)._save_model(epoch, logs)
        if self._max_to_keep is not None:
            directory = self.directory
            checkpoints = [
                fn for fn in os.listdir(self.directory)
                if fn.startswith('model') and fn.endswith('.h5')
            ]
            if len(checkpoints) > self._max_to_keep:
                checkpoints = sorted(checkpoints)
                for checkpoint in checkpoints[:-self._max_to_keep]:
                    os.remove(os.path.join(directory, checkpoint))

    def save_model(self, epoch):
        self._save_model(epoch, logs=None)
