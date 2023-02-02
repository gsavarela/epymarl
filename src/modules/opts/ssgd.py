import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required, _use_grad_for_differentiable
from typing import List, Optional

__all__ = ['SSGD', 'ssgd']


class SSGD(Optimizer):
    def __init__(self, params, lr=required, 
                   *,  foreach: Optional[bool] = None,
                 differentiable=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, foreach=foreach,
                        differentiable=differentiable)
        super(SSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        raise ValueError(f'Sparse Gradients not implemented')

            ssgd(params_with_grad,
                d_p_list,
                lr=group['lr'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])

        return loss



def ssgd(params: List[Tensor],
        d_p_list: List[Tensor],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        lr: float):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd

    func(params,
         d_p_list,
         lr=lr)

def _single_tensor_sgd(params: List[Tensor], d_p_list: List[Tensor], *, lr: float):

    for i, param in enumerate(params):
        d_p = d_p_list[i]

        param.add_(d_p, alpha=-lr)


def _multi_tensor_sgd(
        params: List[Tensor], grads: List[Tensor], *, lr: float):

    if len(params) == 0:
        return

    torch._foreach_add_(params, grads, alpha=-lr)
