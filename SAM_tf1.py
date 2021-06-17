# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Optimizer ops for use in layers and tf.learn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib import framework as contrib_framework
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as vars_
from tensorflow.python.summary import summary
from tensorflow.python.training import moving_averages
from tensorflow.python.training import optimizer as optimizer_
from tensorflow.python.training import training as train


OPTIMIZER_CLS_NAMES = {
    "Adagrad": train.AdagradOptimizer,
    "Adam": train.AdamOptimizer,
    "Ftrl": train.FtrlOptimizer,
    "Momentum": lambda learning_rate: train.MomentumOptimizer(learning_rate, momentum=0.9),  # pylint: disable=line-too-long
    "RMSProp": train.RMSPropOptimizer,
    "SGD": train.GradientDescentOptimizer,
}

OPTIMIZER_SUMMARIES = [
    "learning_rate",
    "loss",
    "gradients",
    "gradient_norm",
    "global_gradient_norm",
]


def optimize_loss_SAM(loss,
                  global_step,
                  learning_rate,
                  optimizer,
                  gradient_noise_scale=None,
                  gradient_multipliers=None,
                  clip_gradients=None,
                  learning_rate_decay_fn=None,
                  update_ops=None,
                  variables=None,
                  name=None,
                  summaries=None,
                  colocate_gradients_with_ops=False,
                  increment_global_step=True):
  """Given loss and parameters for optimizer, returns a training op.
  Various ways of passing optimizers include:
  - by string specifying the name of the optimizer. See OPTIMIZER_CLS_NAMES
      for full list. E.g. `optimize_loss(..., optimizer='Adam')`.
  - by function taking learning rate `Tensor` as argument and returning an
      `Optimizer` instance. E.g. `optimize_loss(...,
      optimizer=lambda lr: tf.compat.v1.train.MomentumOptimizer(lr,
      momentum=0.5))`.
    Alternatively, if `learning_rate` is `None`, the function takes no
    arguments. E.g. `optimize_loss(..., learning_rate=None,
      optimizer=lambda: tf.compat.v1.train.MomentumOptimizer(0.5,
      momentum=0.5))`.
  - by a subclass of `Optimizer` having a single-argument constructor
      (the argument is the learning rate), such as AdamOptimizer or
      AdagradOptimizer. E.g. `optimize_loss(...,
      optimizer=tf.compat.v1.train.AdagradOptimizer)`.
  - by an instance of a subclass of `Optimizer`.
      E.g., `optimize_loss(...,
      optimizer=tf.compat.v1.train.AdagradOptimizer(0.5))`.
  Args:
    loss: Scalar `Tensor`.
    global_step: Scalar int `Tensor`, step counter to update on each step unless
      `increment_global_step` is `False`. If not supplied, it will be fetched
      from the default graph (see `tf.compat.v1.train.get_global_step` for
      details). If it has not been created, no step will be incremented with
      each weight update. `learning_rate_decay_fn` requires `global_step`.
    learning_rate: float or `Tensor`, magnitude of update per each training
      step. Can be `None`.
    optimizer: string, class or optimizer instance, used as trainer. string
      should be name of optimizer, like 'SGD', 'Adam', 'Adagrad'. Full list in
      OPTIMIZER_CLS_NAMES constant. class should be sub-class of `tf.Optimizer`
      that implements `compute_gradients` and `apply_gradients` functions.
      optimizer instance should be instantiation of `tf.Optimizer` sub-class and
      have `compute_gradients` and `apply_gradients` functions.
    gradient_noise_scale: float or None, adds 0-mean normal noise scaled by this
      value.
    gradient_multipliers: dict of variables or variable names to floats. If
      present, gradients for specified variables will be multiplied by given
      constant.
    clip_gradients: float, callable or `None`. If a float is provided, a global
      clipping is applied to prevent the norm of the gradient from exceeding
      this value. Alternatively, a callable can be provided, e.g.,
      `adaptive_clipping_fn()`.  This callable takes a list of `(gradients,
      variables)` tuples and returns the same thing with the gradients modified.
    learning_rate_decay_fn: function, takes `learning_rate` and `global_step`
      `Tensor`s, returns `Tensor`. Can be used to implement any learning rate
      decay functions.
                            For example: `tf.compat.v1.train.exponential_decay`.
                              Ignored if `learning_rate` is not supplied.
    update_ops: list of update `Operation`s to execute at each step. If `None`,
      uses elements of UPDATE_OPS collection. The order of execution between
      `update_ops` and `loss` is non-deterministic.
    variables: list of variables to optimize or `None` to use all trainable
      variables.
    name: The name for this operation is used to scope operations and summaries.
    summaries: List of internal quantities to visualize on tensorboard. If not
      set, the loss, the learning rate, and the global norm of the gradients
      will be reported. The complete list of possible values is in
      OPTIMIZER_SUMMARIES.
    colocate_gradients_with_ops: If True, try colocating gradients with the
      corresponding op.
    increment_global_step: Whether to increment `global_step`. If your model
      calls `optimize_loss` multiple times per training step (e.g. to optimize
      different parts of the model), use this arg to avoid incrementing
      `global_step` more times than necessary.
  Returns:
    Training op.
  Raises:
    ValueError: if:
        * `loss` is an invalid type or shape.
        * `global_step` is an invalid type or shape.
        * `learning_rate` is an invalid type or value.
        * `optimizer` has the wrong type.
        * `clip_gradients` is neither float nor callable.
        * `learning_rate` and `learning_rate_decay_fn` are supplied, but no
          `global_step` is available.
        * `gradients` is empty.
  """
  loss = ops.convert_to_tensor(loss)
  contrib_framework.assert_scalar(loss)
  if global_step is None:
    global_step = train.get_global_step()
  else:
    train.assert_global_step(global_step)
  with vs.variable_scope(name, "OptimizeLoss", [loss, global_step]):
    # Update ops take UPDATE_OPS collection if not provided.
    if update_ops is None:
      update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
    # Make sure update ops are ran before computing loss.
    if update_ops:
      loss = control_flow_ops.with_dependencies(list(update_ops), loss)

    # Learning rate variable, with possible decay.
    lr = None
    if learning_rate is not None:
      if (isinstance(learning_rate, ops.Tensor) and
          learning_rate.get_shape().ndims == 0):
        lr = learning_rate
      elif isinstance(learning_rate, float):
        if learning_rate < 0.0:
          raise ValueError("Invalid learning_rate %s.", learning_rate)
        lr = vs.get_variable(
            "learning_rate", [],
            trainable=False,
            initializer=init_ops.constant_initializer(learning_rate))
      else:
        raise ValueError("Learning rate should be 0d Tensor or float. "
                         "Got %s of type %s" %
                         (str(learning_rate), str(type(learning_rate))))
    if summaries is None:
      summaries = ["loss", "learning_rate", "global_gradient_norm"]
    else:
      for summ in summaries:
        if summ not in OPTIMIZER_SUMMARIES:
          raise ValueError("Summaries should be one of [%s], you provided %s." %
                           (", ".join(OPTIMIZER_SUMMARIES), summ))
    if learning_rate is not None and learning_rate_decay_fn is not None:
      if global_step is None:
        raise ValueError("global_step is required for learning_rate_decay_fn.")
      lr = learning_rate_decay_fn(lr, global_step)
      if "learning_rate" in summaries:
        summary.scalar("learning_rate", lr)

    # Create optimizer, given specified parameters.
    if isinstance(optimizer, six.string_types):
      if lr is None:
        raise ValueError("Learning rate is None, but should be specified if "
                         "optimizer is string (%s)." % optimizer)
      if optimizer not in OPTIMIZER_CLS_NAMES:
        raise ValueError(
            "Optimizer name should be one of [%s], you provided %s." %
            (", ".join(OPTIMIZER_CLS_NAMES), optimizer))
      opt = OPTIMIZER_CLS_NAMES[optimizer](learning_rate=lr)
    elif (isinstance(optimizer, type) and
          issubclass(optimizer, optimizer_.Optimizer)):
      if lr is None:
        raise ValueError("Learning rate is None, but should be specified if "
                         "optimizer is class (%s)." % optimizer)
      opt = optimizer(learning_rate=lr)
    elif isinstance(optimizer, optimizer_.Optimizer):
      opt = optimizer
    elif callable(optimizer):
      if learning_rate is not None:
        opt = optimizer(lr)
      else:
        opt = optimizer()
      if not isinstance(opt, optimizer_.Optimizer):
        raise ValueError("Unrecognized optimizer: function should return "
                         "subclass of Optimizer. Got %s." % str(opt))
    else:
      raise ValueError("Unrecognized optimizer: should be string, "
                       "subclass of Optimizer, instance of "
                       "subclass of Optimizer or function with one argument. "
                       "Got %s." % str(optimizer))

    # All trainable variables, if specific variables are not specified.
    if variables is None:
      variables = vars_.trainable_variables()

    # Compute gradients.
    gradients = opt.compute_gradients(
        loss,
        variables,
        colocate_gradients_with_ops=colocate_gradients_with_ops)
        
    e_ws = []
    rho = constant_op.constant(0.1, dtype=dtypes.float32)
    norm = clip_ops.global_norm(list(zip(*gradients))[0])
    a = rho/(norm + 1e-12)

    for (grad, param) in zip(gradients, variables):
        e_w = grad[0] * a
        param.assign_add(e_w)
        e_ws.append(e_w)

    sam_gradients = opt.compute_gradients(
        loss,
        variables,
        colocate_gradients_with_ops=colocate_gradients_with_ops)

    for (param, e_w) in zip(variables, e_ws):
        param.assign_sub(e_w)

    # Optionally add gradient noise.
    if gradient_noise_scale is not None:
      gradients = _add_scaled_noise_to_gradients(gradients,
                                                 gradient_noise_scale)

    # Multiply some gradients.
    if gradient_multipliers is not None:
      gradients = _multiply_gradients(gradients, gradient_multipliers)
      if not gradients:
        raise ValueError(
            "Empty list of (gradient, var) pairs encountered. This is most "
            "likely to be caused by an improper value of gradient_multipliers.")

    if "global_gradient_norm" in summaries or "gradient_norm" in summaries:
      summary.scalar("global_norm/gradient_norm",
                     clip_ops.global_norm(list(zip(*gradients))[0]))

    # Optionally clip gradients by global norm.
    if isinstance(clip_gradients, float):
      gradients = _clip_gradients_by_norm(gradients, clip_gradients)
    elif callable(clip_gradients):
      gradients = clip_gradients(gradients)
    elif clip_gradients is not None:
      raise ValueError("Unknown type %s for clip_gradients" %
                       type(clip_gradients))

    # Add scalar summary for loss.
    if "loss" in summaries:
      summary.scalar("loss", loss)

    # Add histograms for variables, gradients and gradient norms.
    for gradient, variable in gradients:
      if isinstance(gradient, ops.IndexedSlices):
        grad_values = gradient.values
      else:
        grad_values = gradient

      if grad_values is not None:
        var_name = variable.name.replace(":", "_")
        if "gradients" in summaries:
          summary.histogram("gradients/%s" % var_name, grad_values)
        if "gradient_norm" in summaries:
          summary.scalar("gradient_norm/%s" % var_name,
                         clip_ops.global_norm([grad_values]))

    if clip_gradients is not None and ("global_gradient_norm" in summaries or
                                       "gradient_norm" in summaries):
      summary.scalar("global_norm/clipped_gradient_norm",
                     clip_ops.global_norm(list(zip(*gradients))[0]))

    # Create gradient updates.
    grad_updates = opt.apply_gradients(
        sam_gradients,
        global_step=global_step if increment_global_step else None,
        name="train")

    # Ensure the train_tensor computes grad_updates.
    train_tensor = control_flow_ops.with_dependencies([grad_updates], loss)

    return train_tensor


def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
  """Clips gradients by global norm."""
  gradients, variables = zip(*grads_and_vars)
  clipped_gradients, _ = clip_ops.clip_by_global_norm(gradients, clip_gradients)
  return list(zip(clipped_gradients, variables))


def _adaptive_max_norm(norm, std_factor, decay, global_step, epsilon, name):
  """Find max_norm given norm and previous average."""
  with vs.variable_scope(name, "AdaptiveMaxNorm", [norm]):
    log_norm = math_ops.log(norm + epsilon)

    def moving_average(name, value, decay):
      moving_average_variable = vs.get_variable(
          name,
          shape=value.get_shape(),
          dtype=value.dtype,
          initializer=init_ops.zeros_initializer(),
          trainable=False)
      return moving_averages.assign_moving_average(
          moving_average_variable, value, decay, zero_debias=False)

    # quicker adaptation at the beginning
    if global_step is not None:
      n = math_ops.cast(global_step, dtypes.float32)
      decay = math_ops.minimum(decay, n / (n + 1.))

    # update averages
    mean = moving_average("mean", log_norm, decay)
    sq_mean = moving_average("sq_mean", math_ops.square(log_norm), decay)

    variance = sq_mean - math_ops.square(mean)
    std = math_ops.sqrt(math_ops.maximum(epsilon, variance))
    max_norms = math_ops.exp(mean + std_factor * std)
    return max_norms, mean
