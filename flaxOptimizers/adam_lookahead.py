import numpy as onp
import jax.numpy as jnp
from jax import lax
from flax.optim import OptimizerDef
from flax import struct

@struct.dataclass
class _AdamLookaheadHyperParams:
    learning_rate: onp.ndarray
    beta1: onp.ndarray
    beta2: onp.ndarray
    eps: onp.ndarray
    weight_decay: onp.ndarray
    beta_lookahead: onp.ndarray
    lookahead_every_nth_iter: onp.ndarray

@struct.dataclass
class _AdamLookaheadParamState:
    grad_ema: onp.ndarray
    grad_sq_ema: onp.ndarray
    lookahead_ema: onp.ndarray

class AdamLookahead(OptimizerDef):
    """
    AdamLookahead optimizer, reference implementation used as a template to implement other optimizers.
    https://github.com/google/flax/blob/master/flax/optim/AdamLookahead.py
    """

    def __init__(self, learning_rate=None, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, beta_lookahead=0.5, lookahead_every_nth_iter=5,):
        hyper_params = _AdamLookaheadHyperParams(learning_rate, beta1, beta2, eps, weight_decay, beta_lookahead, lookahead_every_nth_iter)
        super().__init__(hyper_params)

    def init_param_state(self, param):
        return _AdamLookaheadParamState(jnp.zeros_like(param), jnp.zeros_like(param))

    def apply_param_gradient(self, step, hyper_params, param, state, grad):
        assert hyper_params.learning_rate is not None, 'no learning rate provided.'
        beta1 = hyper_params.beta1
        beta2 = hyper_params.beta2
        weight_decay = hyper_params.weight_decay
        grad_sq = lax.square(grad)
        grad_ema = beta1 * state.grad_ema + (1. - beta1) * grad
        grad_sq_ema = beta2 * state.grad_sq_ema + (1. - beta2) * grad_sq

        # bias correction
        t = step + 1.
        grad_ema_corr = grad_ema / (1 - beta1 ** t)
        grad_sq_ema_corr = grad_sq_ema / (1 - beta2 ** t)

        denom = jnp.sqrt(grad_sq_ema_corr) + hyper_params.eps
        new_param = param - hyper_params.learning_rate * grad_ema_corr / denom
        new_param -= hyper_params.learning_rate * weight_decay * param

        # Lookahead
        (new_param, lookahead_ema) = _lookahead(new_param, state.lookahead_ema, t, hyper_params.beta_lookahead, hyper_params.lookahead_every_nth_iter)

        new_state = _AdamLookaheadParamState(grad_ema, grad_sq_ema, lookahead_ema)
        return new_param, new_state
    
def _lookahead(param, lookahead_ema, step, beta_lookahead=0.5, lookahead_every_nth_iter=4):
    """lookahead at the param level instead of group level"""
    condition = step % lookahead_every_nth_iter < 0.5 # == 0. but inexact to deal with roundoffs
    lookahead_ema = jnp.where(condition, beta_lookahead*lookahead_ema + (1. - beta_lookahead)*param, lookahead_ema)
    param = jnp.where(condition, lookahead_ema, param)
    return (param, lookahead_ema)
