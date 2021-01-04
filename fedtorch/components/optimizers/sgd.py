# -*- coding: utf-8 -*-
import torch
from torch.optim.optimizer import Optimizer, required


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, in_momentum=0, out_momentum=0,
                 dampening=0, weight_decay=0, nesterov=False, args=None):
        defaults = dict(lr=lr, in_momentum=in_momentum,
                        out_momentum=out_momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        args=args)
        if nesterov and (in_momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None, apply_lr=True, scale=1.0,
             apply_in_momentum=True, apply_out_momentum=False, **kargs):
        """Performs a single optimization step.

        Avoid to use momentum to accumulate the gradients from other workers.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # retrieve para.
            weight_decay = group['weight_decay']
            in_momentum = group['in_momentum']
            out_momentum = group['out_momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                # add weight decay.
                if weight_decay != 0 and apply_lr:
                    d_p.add_(p.data,alpha=weight_decay)

                # apply local momentum.
                if in_momentum != 0 and apply_in_momentum:
                    if 'in_momentum_buffer' not in param_state:
                        buf = param_state['in_momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(in_momentum).add_(d_p)
                    else:
                        buf = param_state['in_momentum_buffer']
                        buf.mul_(in_momentum).add_(d_p,alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf,alpha=in_momentum)
                    else:
                        d_p = buf

                # apply global momentum.
                if out_momentum != 0 and apply_out_momentum:
                    if 'out_momentum_buffer' not in param_state:
                        buf = param_state['out_momentum_buffer'] = torch.zeros_like(p.grad.data)
                        buf.mul_(out_momentum).add_(d_p)
                    else:
                        buf = param_state['out_momentum_buffer']
                        buf.mul_(out_momentum).add_(d_p,alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf,alpha=out_momentum)
                    else:
                        d_p = buf

                if apply_lr:
                    p.data.add_(d_p,alpha=-group['lr'])
                else:
                    p.data.add_(d_p,alpha=-scale)
        return loss
