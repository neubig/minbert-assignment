from typing import Callable, Iterable, Tuple
import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
  def __init__(
    self,
    params: Iterable[torch.nn.parameter.Parameter],
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-6,
    weight_decay: float = 0.0,
    correct_bias: bool = True,
  ):
    if lr < 0.0:
      raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
    if not 0.0 <= eps:
      raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
    super().__init__(params, defaults)

  def step(self, closure: Callable = None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group["params"]:
        if p.grad is None:
          continue
        grad = p.grad.data
        if grad.is_sparse:
          raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

        # todo
        raise NotImplementedError()

        # Optimizer state should be stored per tensor in `state`
        state = self.state[p]

        # Initialize the state

        # Calculate moments

        state["step"] += 1

        # If this parameter group has its own learning rate, use it,
        # otherwise, use the default.
        step_size = group["lr"]

        if group["correct_bias"]:  # No bias correction for Bert
          bias_correction1 = 1.0 - beta1 ** state["step"]
          bias_correction2 = 1.0 - beta2 ** state["step"]
          step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

        # Update parameters

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't interact
        # with the m/v parameters. This is equivalent to adding the square
        # of the weights to the loss with plain (non-momentum) SGD.
        # Add weight decay at the end (fixed version)
        if group["weight_decay"] > 0.0:
          pass

    return loss
