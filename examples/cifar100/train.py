from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import tensorflow_datasets as tfds

import more_keras.callbacks as cb
from more_keras.framework.train import train

import parts

epochs = 3
model_fn = parts.get_cifar_model
train(problem=parts.problem,
      train_pipeline=parts.train_pipeline,
      validation_pipeline=parts.validation_pipeline,
      model_fn=parts.get_cifar_model,
      optimizer=parts.optimizer,
      epochs=epochs,
      chkpt_dir=parts.chkpt_dir)
