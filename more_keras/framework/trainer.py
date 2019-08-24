from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
import os
import gin
import tensorflow as tf
from more_keras.framework import pipelines as pl
from more_keras.framework import problems as prob
from more_keras import callbacks as cb


def _dataset_and_model_simple(problem, pipeline, model_fn, split):
    """
    Get dataset(s) and model for basic supervised training.

    Args:
        problem: `Problem` instance.
        pipeline: `Pipeline` instance of list/tuple of instances. Structure
            should match that of `split`.
        model_fn: function taking (input_spec, output_spec) and returning
            a keras model.
        split: string ("train", "validation", "test") or list/tuple of strings.
            Structure should match that of `pipeline`.

    Returns:
        dataset: preprocessed dataset, or list/tuple of datasets with structure
            matching that of split / pipeline.
        model: keras model.
    """
    dataset = problem.get_base_dataset(split)
    dataset = tf.nest.map_structure(lambda ds, pipeline: pipeline(ds), dataset,
                                    pipeline)

    first_pipeline = tf.nest.flatten(pipeline)[0]
    model = model_fn(first_pipeline.output_spec(problem.input_spec),
                     problem.output_spec)
    return dataset, model


@gin.configurable(module='mk.framework')
class Trainer(object):

    def __init__(self, problem, pipeline, model_fn, optimizer_fn):
        """
        Args:
            problem: `framework.problems.Problem` instance.
            pipeline: `framework.pipelines.Pipeline` instances or dict with
                instance values and at least a 'train' and 'validation' or
                'test' keys.
            model_fn: function mapping (input_spec, output_spec) -> model.
            optimizer_fn: function mapping () -> optimizer.
        """
        self._problem = problem
        if isinstance(pipeline, pl.Pipeline):
            pipelines = {
                'train': pipeline,
                'validation': pipeline,
                'test': pipeline,
            }
        elif isinstance(pipeline, dict):
            pipelines = pipeline.copy()
            for k, v in pipelines.items():
                if k not in ('train', 'test', 'validation'):
                    raise KeyError('Invalid key for pipelines {} - must be in '
                                   '("train", "test", "validation")'.format(k))
                if not isinstance(v, pl.Pipeline):
                    raise ValueError(
                        'pipelines must all be Pipeline instances, '
                        'got {} for split {}'.format(v, k))
            if 'train' not in pipelines:
                raise ValueError(
                    'pipelines must have "train" key if supplied as a dict')
            if len(pipelines) == 1:
                # only train - copy across to validation/splits
                pipelines['test'] = pipelines['validation'] = pipelines['train']
            else:
                # copy test -> evaluation or vice versa if either is not present
                if 'test' not in pipelines and 'validation' in pipelines:
                    pipelines['test'] = pipelines['validation']
                if 'validation' not in pipelines and 'test' in pipelines:
                    pipelines['validation'] = pipelines['test']
        else:
            raise TypeError(
                '`pipelines` must be a `Pipeline` instance or dict, got {}'.
                format(pipelines))
        self._pipelines = pipelines

        if not callable(model_fn):
            raise ValueError(
                'model_fn must be callable, got {}'.format(model_fn))
        if isinstance(model_fn, tf.keras.models.Model):
            raise TypeError('model_fn should be a callable producing a model, '
                            'not a model itself, got {}'.format(model_fn))
        self._model_fn = model_fn

        if not callable(optimizer_fn):
            raise ValueError(
                'optimizer_fn must be callable, got {}'.format(optimizer_fn))
        self._optimizer_fn = optimizer_fn

        self._tensorboard = cb.tensorboard.BetterTensorBoard(None)
        self._model = None
        self._optimizer = None
        self._callbacks = []

    def _rebuild_model(self, input_spec, output_spec):
        self._model = self._model_fn(self.model_input_spec, self.output_spec)
        self.compile_model()

    def rebuild_model(self):
        with cb.aggregator.Aggregator() as callback_agg:
            with cb.cache.Cache() as cache:
                with self._tensorboard:
                    self._rebuild_model(self.model_input_spec, self.output_spec)
                if len(cache) > 0:
                    callback_agg.append(cache)
        self._callbacks = callback_agg.callbacks

    def rebuild_optimizer(self):
        self._optimizer = self._optimizer_fn()

    @property
    def problem(self):
        return self._problem

    @property
    def model(self):
        if self._model is None:
            self.rebuild_model()
        return self._model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self.rebuild_optimizer()
        return self._optimizer

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.problem.loss,
                           metrics=self.problem.metrics)

    def get_pipeline(self, split):
        return self._pipelines[split]

    def steps_per_epoch(self, split):
        return self.problem.examples_per_epoch(split) // self.get_pipeline(
            split).batch_size

    def get_dataset(self, split):
        return self.get_pipeline(split)(self.problem.get_base_dataset(split))

    @property
    def model_input_spec(self):
        return self.get_pipeline('train').output_spec(self.problem.input_spec)

    @property
    def output_spec(self):
        return self.problem.output_spec

    def _saver_callback(self, model_dir):
        return cb.BetterModelCheckpoint(model_dir, load_weights_on_restart=True)

    def train(self,
              epochs,
              model_dir,
              save_gin_config=True,
              verbose=True,
              fresh=False,
              extra_callbacks=None,
              train_steps=None,
              validation_steps=None):
        if model_dir is not None:
            model_dir = os.path.expanduser(os.path.expandvars(model_dir))

        callbacks = list(self._callbacks)
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())

        if fresh and tf.io.gfile.exists(model_dir):
            tf.io.gfile.rmtree(model_dir)

        if model_dir is None:
            initial_epoch = 0
        else:
            if not tf.io.gfile.isdir(model_dir):
                tf.io.gfile.makedirs(model_dir)
            chkpt_callback = self._saver_callback(model_dir)
            chkpt = chkpt_callback.checkpoint()
            if chkpt is None:
                initial_epoch = 0
            else:
                initial_epoch = chkpt_callback.epoch(chkpt)
            callbacks.append(chkpt_callback)

        if train_steps is None:
            train_steps = self.steps_per_epoch('train')
        if validation_steps is None:
            validation_steps = self.steps_per_epoch('validation')

        # this will be logged anyway, but sometimes is nice to have separately
        if save_gin_config and model_dir is not None:
            callbacks.append(cb.GinConfigSaver(model_dir))

        # user supplied extra callbacks
        if extra_callbacks is not None:
            callbacks.extend(extra_callbacks)

        # put this last so we log any updates from other callbacks
        if model_dir is not None:
            self._tensorboard.log_dir = model_dir
            callbacks.append(self._tensorboard)

        # log operative config and model summary
        logging.info('Training starting with operative config: \n{}'.format(
            gin.operative_config_str()))
        self.model.summary(print_fn=logging.info)

        history = self.model.fit(
            self.get_dataset('train'),
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=self.get_dataset('validation'),
            steps_per_epoch=train_steps,
            validation_steps=validation_steps,
            initial_epoch=initial_epoch,
        )
        return history

    def evaluate(self,
                 model_dir,
                 steps=None,
                 verbose=True,
                 extra_callbacks=None):
        if steps is None:
            steps = self.steps_per_epoch('validation')
        callbacks = list(self._callbacks)
        if model_dir is not None:
            callbacks.append(self._saver_callback(model_dir))
        if extra_callbacks is not None:
            callbacks.extend(extra_callbacks)
        return self.model.evaluate(self.get_dataset('validation'),
                                   steps=steps,
                                   callbacks=callbacks,
                                   verbose=verbose)


@gin.configurable(module='mk.framework')
class MetaTrainer(Trainer):

    def _saver_callback(self, chkpt_dir):
        # Seems to be a keras bug rearing its head during self._model.get_config
        # so we use a tensorflow-based saver
        return cb.SaverCallback(chkpt_dir, load_weights_on_restart=True)

    def _rebuild_model(self, input_spec, output_spec):
        from more_keras.meta_models import builder as b
        with b.MetaNetworkBuilder() as builder:
            self._model = self._model_fn(input_spec, output_spec)
            self.compile_model()

            # build preprocessor
            labels_spec = self.problem.labels_spec
            weights_spec = self.problem.weights_spec
            labels, weights = tf.nest.map_structure(
                lambda spec: None if spec is None else builder.batched(
                    builder.prebatch_input(shape=spec.shape, dtype=spec.dtype)),
                (labels_spec, weights_spec))
            self._preprocessor = builder.preprocessor(labels, weights)

    def get_dataset(self, split):
        pipeline = self.get_pipeline(split)
        dataset = self.problem.get_base_dataset(split)
        if pipeline.shuffle_buffer is not None:
            dataset = dataset.shuffle(pipeline.shuffle_buffer)
        if pipeline.repeats != pl.NO_REPEAT:
            dataset = dataset.repeat(pipeline.repeats)
        if pipeline.map_fn is not None:
            dataset = dataset.map(
                pipeline.map_fn, num_parallel_calls=pipeline.num_parallel_calls)
        dataset = self._preprocessor.map_and_batch(
            dataset,
            batch_size=pipeline.batch_size,
            num_parallel_calls=pipeline.num_parallel_calls)
        if pipeline.prefetch_buffer:
            dataset = dataset.prefetch(pipeline.prefetch_buffer)

        return dataset


@gin.configurable(module='mk.framework')
def train(trainer,
          epochs,
          model_dir,
          save_gin_config=True,
          verbose=True,
          fresh=False,
          extra_callbacks=None,
          train_steps=None,
          validation_steps=None):
    return trainer.train(
        epochs=epochs,
        model_dir=model_dir,
        save_gin_config=save_gin_config,
        verbose=verbose,
        fresh=fresh,
        extra_callbacks=extra_callbacks,
        train_steps=train_steps,
        validation_steps=validation_steps,
    )


@gin.configurable(module='mk.framework')
def evaluate(
        trainer,
        model_dir,
        steps=None,
        verbose=True,
        extra_callbacks=None,
):
    trainer.evaluate(
        model_dir=model_dir,
        steps=steps,
        verbose=verbose,
        extra_callbacks=extra_callbacks,
    )
