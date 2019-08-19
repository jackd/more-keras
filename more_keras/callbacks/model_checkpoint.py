from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import gin
LATEST = 'LATEST'


@gin.configurable(module='mk.callbacks')
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

    def reset_state(self):
        self._started = False

    def on_train_begin(self, logs=None):
        self._started = True
        out = super(BetterModelCheckpoint, self).on_train_begin(logs=logs)
        if self.load_weights_on_restart:
            self.restore_optimizer()
        return out

    def on_test_begin(self, logs=None):
        if not self._started and self.load_weights_on_restart:
            self.restore_model()

    def on_predict_begin(self, logs=None):
        if not self._started and self.load_weights_on_restart:
            self.restore_model()

    @property
    def directory(self):
        return self._directory

    @property
    def latest_checkpoint(self):
        return self._get_most_recently_modified_file_matching_pattern(
            self.filepath)

    def checkpoint(self, checkpoint_or_epoch_or_latest):
        if checkpoint_or_epoch_or_latest is None:
            return None
        elif checkpoint_or_epoch_or_latest == LATEST:
            return self.latest_checkpoint
        elif isinstance(checkpoint_or_epoch_or_latest, int):
            return self.filepath.format(epoch=checkpoint_or_epoch_or_latest)
        elif tf.io.gfile.exists(checkpoint_or_epoch_or_latest):
            return checkpoint_or_epoch_or_latest
        else:
            raise ValueError(
                'Unrecognized value for checkpoint_or_epoch_or_latest, {}'.
                format(checkpoint_or_epoch_or_latest))

    @property
    def latest_epoch(self):
        checkpoint = self.latest_checkpoint
        return None if checkpoint is None else self.epoch(checkpoint)

    def epoch(self, checkpoint):
        return int(checkpoint[-8:-3])

    def restore_model(self, checkpoint=LATEST):
        """Does not restore optimizer weights. See `restore_optimizer`."""
        chkpt = self.checkpoint(checkpoint)
        if chkpt is None:
            return
        self.model.load_weights(self.checkpoint(checkpoint))

    def restore_optimizer(self, checkpoint=LATEST):
        """Restore weights to optimizer."""
        # pylint: disable=no-name-in-module
        from tensorflow.python.keras.saving.hdf5_format import \
            load_optimizer_weights_from_hdf5_group
        # pylint: enable=no-name-in-module
        import h5py
        checkpoint = self.checkpoint(checkpoint)
        if checkpoint is None:
            return
        with h5py.File(checkpoint) as f:
            optimizer_weight_values = load_optimizer_weights_from_hdf5_group(f)
            self.model.optimizer.set_weights(optimizer_weight_values)

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
