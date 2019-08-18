from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from more_keras.framework.problems.tfds import TfdsProblem
from more_keras.framework.pipelines import Pipeline
from more_keras.framework.train import evaluate

import os
import parts

evaluate(problem=parts.problem,
         validation_pipeline=parts.validation_pipeline,
         model_fn=parts.get_cifar_model,
         optimizer=parts.optimizer,
         chkpt_dir=parts.chkpt_dir)
