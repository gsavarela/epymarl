import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required, _use_grad_for_differentiable 
from typing import List, Optional
from copy import deepcopy
from IPython.core.debugger import set_trace

__all__ = ['CSGD', 'csgd']


class CSGD(Optimizer):
    def __init__(self, local_params, consensus_params, lr=required,
                   *,  foreach: Optional[bool] = None,
                 differentiable=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, foreach=foreach,
                        differentiable=differentiable)
        params = [{'params': local_params}, {'params': consensus_params}]
        super(CSGD, self).__init__(params, defaults)
        self.state['prev_consensus_params'] = deepcopy(consensus_params)
        

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

        local = self.param_groups[0]['params']
        consensus = self.param_groups[-1]['params']
        prev = self.state['prev_consensus_params']
        params_with_grad = []
        d_p_list = []
        consensus_update = False
        for lcl, cns, prv in zip(local, consensus, prev):
            assert lcl.shape == cns.shape
            if lcl.grad is not None:
                consensus_update |= not torch.allclose(cns, prv)
                if consensus_update:  # Update
                    params_with_grad.append(cns)
                else:
                    params_with_grad.append(lcl)
                d_p_list.append(lcl.grad)
                if lcl.grad.is_sparse:
                    raise ValueError(f'Sparse Gradients not implemented')

                csgd(params_with_grad, d_p_list,
                     lr=self.defaults['lr'], foreach=self.defaults['foreach'])

                if consensus_update:  # Update
                    # Copy parameters from local to consensus
                    lcl.data = cns.data
        if consensus_update:
            self.state['prev_consensus_params'] = deepcopy(consensus)

        return loss

def csgd(params: List[Tensor],
        d_p_list: List[Tensor],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
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
    set_trace()
    torch._foreach_add_(params, grads, alpha=-lr)
