# Copyright 2021 The Flax Authors.
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

from .. import struct

from jax import lax
import jax.numpy as jnp

import numpy as np

from .base import OptimizerDef

@struct.dataclass
class _LAMBLookaheadHyperParams:
  learning_rate: np.ndarray
  beta1: np.ndarray
  beta2: np.ndarray
  weight_decay: np.ndarray
  eps: np.ndarray
  beta_lookahead: np.ndarray
  lookahead_every_nth_iter: np.ndarray


@struct.dataclass
class _LAMBLookaheadParamState:
  grad_ema: np.ndarray
  grad_sq_ema: np.ndarray
  lookahead_ema: np.ndarray


class LAMBLookahead(OptimizerDef):
  """Layerwise adaptive moments for batch (LAMBLookahead) optimizer.

  See https://arxiv.org/abs/1904.00962
  """

  def __init__(self, learning_rate=None, beta1=0.9, beta2=0.999, weight_decay=0,
               eps=1e-6):
    """Constructor for the LAMBLookahead optimizer.

    Args:
      learning_rate: the step size used to update the parameters.
      beta1: the coefficient used for the moving average of the gradient
        (default: 0.9).
      beta2: the coefficient used for the moving average of the squared gradient
        (default: 0.999).
      weight_decay: weight decay coefficient to apply
      eps: epsilon used for Adam update computation (default: 1e-6).
    """

    hyper_params = _LAMBLookaheadHyperParams(
        learning_rate, beta1, beta2, weight_decay, eps)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    return _LAMBLookaheadParamState(jnp.zeros_like(param), jnp.zeros_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    beta1 = hyper_params.beta1
    beta2 = hyper_params.beta2
    weight_decay = hyper_params.weight_decay
    learning_rate = hyper_params.learning_rate

    grad_sq = lax.square(grad)
    grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
    grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

    t = jnp.array(step + 1, lax.dtype(param.dtype))
    grad_ema_corr = grad_ema / (1. - beta1 ** t)
    grad_sq_ema_corr = grad_sq_ema / (1. - beta2 ** t)

    update = grad_ema_corr / (jnp.sqrt(grad_sq_ema_corr) + hyper_params.eps)

    if weight_decay != 0.0:
      update += weight_decay * param

    param_norm = jnp.linalg.norm(param)
    update_norm = jnp.linalg.norm(update)
    trust_ratio = jnp.where(
        param_norm + update_norm > 0., param_norm / update_norm, 1.)

    new_param = param - trust_ratio * learning_rate * update

    # Lookahead
    (new_param, lookahead_ema) = _lookahead(new_param, state.lookahead_ema, t, hyper_params.beta_lookahead, hyper_params.lookahead_every_nth_iter)

    new_state = _LAMBLookaheadParamState(grad_ema, grad_sq_ema, lookahead_ema)

    return new_param, new_state


def _lookahead(param, lookahead_ema, step, beta_lookahead=0.5, lookahead_every_nth_iter=4):
    """lookahead at the param level instead of group level"""
    condition = step % lookahead_every_nth_iter < 0.5 # == 0. but inexact to deal with roundoffs
    lookahead_ema = jnp.where(condition, beta_lookahead*lookahead_ema + (1. - beta_lookahead)*param, lookahead_ema)
    param = jnp.where(condition, lookahead_ema, param)
    return (param, lookahead_ema)