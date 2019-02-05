# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Helper wrapper for a Tensorflow optimizer."""

import numpy as np
import tensorflow as tf

from collections import OrderedDict
from typing import List, Union

from . import autosummary
from . import tfutil
from .. import util

from .tfutil import TfExpression, TfExpressionEx

try:
    # TensorFlow 1.13
    from tensorflow.python.ops import nccl_ops
except:
    # Older TensorFlow versions
    import tensorflow.contrib.nccl as nccl_ops

class Optimizer:
    """A Wrapper for tf.train.Optimizer.

    Automatically takes care of:
    - Gradient averaging for multi-GPU training.
    - Dynamic loss scaling and typecasts for FP16 training.
    - Ignoring corrupted gradients that contain NaNs/Infs.
    - Reporting statistics.
    - Well-chosen default settings.
    """

    def __init__(self,
                 name: str = "Train",
                 tf_optimizer: str = "tf.train.AdamOptimizer",
                 learning_rate: TfExpressionEx = 0.001,
                 use_loss_scaling: bool = False,
                 loss_scaling_init: float = 64.0,
                 loss_scaling_inc: float = 0.0005,
                 loss_scaling_dec: float = 1.0,
                 **kwargs):

        # Init fields.
        self.name = name
        self.learning_rate = tf.convert_to_tensor(learning_rate)
        self.id = self.name.replace("/", ".")
        self.scope = tf.get_default_graph().unique_name(self.id)
        self.optimizer_class = util.get_obj_by_name(tf_optimizer)
        self.optimizer_kwargs = dict(kwargs)
        self.use_loss_scaling = use_loss_scaling
        self.loss_scaling_init = loss_scaling_init
        self.loss_scaling_inc = loss_scaling_inc
        self.loss_scaling_dec = loss_scaling_dec
        self._grad_shapes = None  # [shape, ...]
        self._dev_opt = OrderedDict()  # device => optimizer
        self._dev_grads = OrderedDict()  # device => [[(grad, var), ...], ...]
        self._dev_ls_var = OrderedDict()  # device => variable (log2 of loss scaling factor)
        self._updates_applied = False

    def register_gradients(self, loss: TfExpression, trainable_vars: Union[List, dict]) -> None:
        """Register the gradients of the given loss function with respect to the given variables.
        Intended to be called once per GPU."""
        assert not self._updates_applied

        # Validate arguments.
        if isinstance(trainable_vars, dict):
            trainable_vars = list(trainable_vars.values())  # allow passing in Network.trainables as vars

        assert isinstance(trainable_vars, list) and len(trainable_vars) >= 1
        assert all(tfutil.is_tf_expression(expr) for expr in trainable_vars + [loss])

        if self._grad_shapes is None:
            self._grad_shapes = [tfutil.shape_to_list(var.shape) for var in trainable_vars]

        assert len(trainable_vars) == len(self._grad_shapes)
        assert all(tfutil.shape_to_list(var.shape) == var_shape for var, var_shape in zip(trainable_vars, self._grad_shapes))

        dev = loss.device

        assert all(var.device == dev for var in trainable_vars)

        # Register device and compute gradients.
        with tf.name_scope(self.id + "_grad"), tf.device(dev):
            if dev not in self._dev_opt:
                opt_name = self.scope.replace("/", "_") + "_opt%d" % len(self._dev_opt)
                assert callable(self.optimizer_class)
                self._dev_opt[dev] = self.optimizer_class(name=opt_name, learning_rate=self.learning_rate, **self.optimizer_kwargs)
                self._dev_grads[dev] = []

            loss = self.apply_loss_scaling(tf.cast(loss, tf.float32))
            grads = self._dev_opt[dev].compute_gradients(loss, trainable_vars, gate_gradients=tf.train.Optimizer.GATE_NONE)  # disable gating to reduce memory usage
            grads = [(g, v) if g is not None else (tf.zeros_like(v), v) for g, v in grads]  # replace disconnected gradients with zeros
            self._dev_grads[dev].append(grads)

    def apply_updates(self) -> tf.Operation:
        """Construct training op to update the registered variables based on their gradients."""
        tfutil.assert_tf_initialized()
        assert not self._updates_applied
        self._updates_applied = True
        devices = list(self._dev_grads.keys())
        total_grads = sum(len(grads) for grads in self._dev_grads.values())
        assert len(devices) >= 1 and total_grads >= 1
        ops = []

        with tfutil.absolute_name_scope(self.scope):
            # Cast gradients to FP32 and calculate partial sum within each device.
            dev_grads = OrderedDict()  # device => [(grad, var), ...]

            for dev_idx, dev in enumerate(devices):
                with tf.name_scope("ProcessGrads%d" % dev_idx), tf.device(dev):
                    sums = []

                    for gv in zip(*self._dev_grads[dev]):
                        assert all(v is gv[0][1] for g, v in gv)
                        g = [tf.cast(g, tf.float32) for g, v in gv]
                        g = g[0] if len(g) == 1 else tf.add_n(g)
                        sums.append((g, gv[0][1]))

                    dev_grads[dev] = sums

            # Sum gradients across devices.
            if len(devices) > 1:
                with tf.name_scope("SumAcrossGPUs"), tf.device(None):
                    for var_idx, grad_shape in enumerate(self._grad_shapes):
                        g = [dev_grads[dev][var_idx][0] for dev in devices]

                        if np.prod(grad_shape):  # nccl does not support zero-sized tensors
                            g = nccl_ops.all_sum(g)

                        for dev, gg in zip(devices, g):
                            dev_grads[dev][var_idx] = (gg, dev_grads[dev][var_idx][1])

            # Apply updates separately on each device.
            for dev_idx, (dev, grads) in enumerate(dev_grads.items()):
                with tf.name_scope("ApplyGrads%d" % dev_idx), tf.device(dev):
                    # Scale gradients as needed.
                    if self.use_loss_scaling or total_grads > 1:
                        with tf.name_scope("Scale"):
                            coef = tf.constant(np.float32(1.0 / total_grads), name="coef")
                            coef = self.undo_loss_scaling(coef)
                            grads = [(g * coef, v) for g, v in grads]

                    # Check for overflows.
                    with tf.name_scope("CheckOverflow"):
                        grad_ok = tf.reduce_all(tf.stack([tf.reduce_all(tf.is_finite(g)) for g, v in grads]))

                    # Update weights and adjust loss scaling.
                    with tf.name_scope("UpdateWeights"):
                        # pylint: disable=cell-var-from-loop
                        opt = self._dev_opt[dev]
                        ls_var = self.get_loss_scaling_var(dev)

                        if not self.use_loss_scaling:
                            ops.append(tf.cond(grad_ok, lambda: opt.apply_gradients(grads), tf.no_op))
                        else:
                            ops.append(tf.cond(grad_ok,
                                               lambda: tf.group(tf.assign_add(ls_var, self.loss_scaling_inc), opt.apply_gradients(grads)),
                                               lambda: tf.group(tf.assign_sub(ls_var, self.loss_scaling_dec))))

                    # Report statistics on the last device.
                    if dev == devices[-1]:
                        with tf.name_scope("Statistics"):
                            ops.append(autosummary.autosummary(self.id + "/learning_rate", self.learning_rate))
                            ops.append(autosummary.autosummary(self.id + "/overflow_frequency", tf.where(grad_ok, 0, 1)))

                            if self.use_loss_scaling:
                                ops.append(autosummary.autosummary(self.id + "/loss_scaling_log2", ls_var))

            # Initialize variables and group everything into a single op.
            self.reset_optimizer_state()
            tfutil.init_uninitialized_vars(list(self._dev_ls_var.values()))

            return tf.group(*ops, name="TrainingOp")

    def reset_optimizer_state(self) -> None:
        """Reset internal state of the underlying optimizer."""
        tfutil.assert_tf_initialized()
        tfutil.run([var.initializer for opt in self._dev_opt.values() for var in opt.variables()])

    def get_loss_scaling_var(self, device: str) -> Union[tf.Variable, None]:
        """Get or create variable representing log2 of the current dynamic loss scaling factor."""
        if not self.use_loss_scaling:
            return None

        if device not in self._dev_ls_var:
            with tfutil.absolute_name_scope(self.scope + "/LossScalingVars"), tf.control_dependencies(None):
                self._dev_ls_var[device] = tf.Variable(np.float32(self.loss_scaling_init), name="loss_scaling_var")

        return self._dev_ls_var[device]

    def apply_loss_scaling(self, value: TfExpression) -> TfExpression:
        """Apply dynamic loss scaling for the given expression."""
        assert tfutil.is_tf_expression(value)

        if not self.use_loss_scaling:
            return value

        return value * tfutil.exp2(self.get_loss_scaling_var(value.device))

    def undo_loss_scaling(self, value: TfExpression) -> TfExpression:
        """Undo the effect of dynamic loss scaling for the given expression."""
        assert tfutil.is_tf_expression(value)

        if not self.use_loss_scaling:
            return value

        return value * tfutil.exp2(-self.get_loss_scaling_var(value.device)) # pylint: disable=invalid-unary-operand-type
