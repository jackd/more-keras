from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from more_keras import callbacks as cb


def dataset_and_steps(problem, split):
    return tf.nest.map_structure(tuple, problem.get_base_dataset(split),
                                 problem.examples_per_epoch(split))


def train(problem,
          train_pipeline,
          validation_pipeline,
          model_fn,
          optimizer,
          epochs,
          chkpt_dir=None,
          log_dir=None,
          fresh=False,
          verbose=True,
          extra_callbacks=None,
          train_steps=None,
          validation_steps=None):
    with cb.aggregator.Aggregator() as callback_agg:
        with cb.cache.Cache() as cache:
            train_ds, val_ds = problem.get_base_dataset(('train', 'validation'))
            train_ds = train_pipeline(train_ds)
            val_ds = validation_pipeline(val_ds)
            model = model_fn(train_pipeline.output_spec, problem.output_spec)
            model.compile(optimizer=optimizer,
                          loss=problem.loss,
                          metrics=problem.metrics)
            if len(cache) > 0:
                callback_agg.append(cache)
    callbacks = callback_agg.callbacks
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())

    if chkpt_dir is not None:
        chkpt_dir = os.path.expanduser(os.path.expandvars(chkpt_dir))
    if log_dir is None:
        log_dir = chkpt_dir
    if log_dir is not None:
        log_dir = os.path.expanduser(os.path.expandvars(log_dir))

    if fresh:
        for d in log_dir, chkpt_dir:
            tf.io.gfile.rmtree(d)
            tf.io.gfile.makedirs(d)
    else:
        for d in log_dir, chkpt_dir:
            if not tf.io.gfile.exists(d):
                tf.io.gfile.makedirs(d)

    if chkpt_dir is None:
        initial_epoch = 0
    else:
        if not tf.io.gfile.isdir(chkpt_dir):
            tf.io.gfile.makedirs(chkpt_dir)
        chkpt_callback = cb.BetterModelCheckpoint(chkpt_dir,
                                                  load_weights_on_restart=True)
        chkpt = chkpt_callback.latest_checkpoint
        if chkpt is None:
            initial_epoch = 0
        else:
            initial_epoch = chkpt_callback.epoch(chkpt)
        callbacks.append(chkpt_callback)

    if log_dir is not None:
        log_dir = os.path.expanduser(os.path.expandvars(log_dir))
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir))

    if train_steps is None:
        train_steps = problem.examples_per_epoch(
            'train') // train_pipeline.batch_size
    if validation_steps is None:
        validation_steps = problem.examples_per_epoch(
            'validation') // validation_pipeline.batch_size

    callbacks.extend(extra_callbacks)

    history = model.fit(
        train_ds,
        epochs=epochs,
        verbose=verbose,
        callbacks=list(callbacks),
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=validation_steps,
        initial_epoch=initial_epoch,
    )
    return history


def evaluate(problem,
             validation_pipeline,
             model_fn,
             optimizer,
             chkpt_dir,
             verbose=True):
    with cb.aggregator.Aggregator() as callback_agg:
        with cb.cache.Cache() as cache:
            val_ds = validation_pipeline(problem.get_base_dataset('validation'))
            model = model_fn(validation_pipeline.output_spec,
                             problem.output_spec)
            model.compile(optimizer=optimizer,
                          loss=problem.loss,
                          metrics=problem.metrics)
            if len(cache) > 0:
                callback_agg.append(cache)
    callbacks = callback_agg.callbacks
    chkpt_callback = cb.BetterModelCheckpoint(chkpt_dir,
                                              load_weights_on_restart=True)

    callbacks.append(chkpt_callback)
    steps = problem.examples_per_epoch(
        'validation') // validation_pipeline.batch_size

    history = model.evaluate(val_ds,
                             steps=steps,
                             callbacks=callbacks,
                             verbose=verbose)
    return history
