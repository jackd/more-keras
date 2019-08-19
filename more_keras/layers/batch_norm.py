from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
import tensorflow as tf


@gin.configurable(module='mk.layers')
class VariableMomentumBatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Slight modification to BatchNormalization that allows for variable momentum.

    This momentum is not trainable but can be updated directly e.g. with
    callbacks.

    Example usage:
    ```
    import numpy as np
    import functools
    from more_keras.callbacks import ScheduleUpdater
    from more_keras.schedules import complementary_exponential_decay
    updater = ScheduleUpdater(
        functools.partial(
            complementary_exponential_decay,
            initial_value=0.5,
            decay_rate=0.5,
            decay_steps=10,
            asumptote=1,
            clip_value=0.99,
            impl=np),
        variables_func=lamba model: [
            layer.momentum for layer in model.layers if isinstance(
                layer, VariableMomentumBatchNormalization)]
        )
    model = get_model(...)
    model.fit(data, callbacks=[updater])
    """

    def build(self, input_shape):
        if not self.built:
            self.initial_momentum = self.momentum
            self.momentum = self.add_weight(
                'momentum',
                initializer=tf.keras.initializers.constant(self.momentum),
                # initializer=tf.keras.initializers.constant(
                #     self.momentum * np.ones(shape=input_shape[-1])),
                trainable=False)
            super(VariableMomentumBatchNormalization, self).build(input_shape)

    def get_config(self):
        config = super(VariableMomentumBatchNormalization, self).get_config()
        if self.built:
            # actual momentum is a variable and should be saved in weights
            config['momentum'] = self.initial_momentum
        return config
