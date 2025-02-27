import torch
import torch.nn as nn


# def euler_step(o: torch.Tensor, do: torch.Tensor, step_size: float,
#                decay: float | None = None, in_place: bool = False) -> torch.Tensor:
#     """Euler step, vanilla case."""
#     if not in_place:
#         if decay is None or decay == 0.:
#             oo = o + step_size * do
#         else:
#             # todo togli step_size da moltiplicare decay
#             oo = (1. + step_size * decay) * o + step_size * do
#         return oo
#     else:
#         if decay is None or decay == 0.:
#             o.add_(do, alpha=step_size)
#         else:
#             o.mul_(1. + step_size * decay).add_(do, alpha=step_size)
#         return o


def _euler_step(o: torch.Tensor | dict[str, torch.Tensor], do: torch.Tensor | dict[str, torch.Tensor],
                step_size: float, decay: float | None = None, in_place: bool = False) \
        -> torch.Tensor | dict[str, torch.Tensor]:
    """Euler step, vanilla case.

    Params:
        o: list or dict of data to update (warning: it will be updated here).
        do: list or dict of derivatives w.r.t. time of the data we want to update.
        step_size: the step size of the Euler method.
        decay: the weight-decay-like scalar coefficient that tunes the strength of the weight-decay regularization.
        in_place: whether to overwrite the input data or to return a new list with new elements.

    Returns:
        A list or dict of Tensors with the same size of the input 'o'. It could be a new list with new Tensors
        (if in_place is False) or 'o' itself, updated in-place (if in_place is True).
    """
    assert type(o) is type(do), f'Input should either be two lists or two dicts, got {type(o)} and {type(do)}.'

    if isinstance(o, dict):
        assert set(o.keys()) == set(do.keys()), 'Dictionaries should have the same keys.'
        oo = dict.fromkeys(o)
        for k in o.keys():
            if not in_place:
                if decay is None or decay == 0.:
                    oo[k] = o[k] + step_size * do[k]
                else:
                    oo[k] = (1. + decay) * o[k] + step_size * do[k]
                return oo
            else:
                if decay is None or decay == 0.:
                    o[k].add_(do[k], alpha=step_size)
                else:
                    o[k].mul_(1. + decay).add_(do[k], alpha=step_size)
                return o
    elif isinstance(o, torch.Tensor):
        if not in_place:
            if decay is None or decay == 0.:
                oo = o + step_size * do
            else:
                oo = (1. + decay) * o + step_size * do
            return oo
        else:
            if decay is None or decay == 0.:
                o.add_(do, alpha=step_size)
            else:
                o.mul_(1. + decay).add_(do, alpha=step_size)
            return o
    else:
        raise Exception(f'Input to this function should be either list or dict, got {type(o)}.')


def _init(val: float | str, data_shape: torch.Size, device, dtype: torch.dtype, non_negative: bool = False) -> torch.Tensor:
    """Initialize a tensor to a constant value or to random values, or to zeros (and possibly others).

    Params:
        val: a float value or a string in ['random', 'zeros'].
        data_shape: the shape of the target tensor.
        device: the device where the tensor will be stored.
        non_negative: whether to create something non-negative.

    Returns:
        An initialized tensor.
    """
    assert type(val) is float or val in ['zeros', 'random', 'ones', 'alternating'], 'Invalid initialization: ' + str(val)

    if isinstance(val, float):
        t = torch.full(data_shape, val, device=device, dtype=dtype)
        if non_negative:
            t = torch.abs(t)
        return t
    elif val == 'random':
        t = torch.randn(data_shape, device=device, dtype=dtype)
        if non_negative:
            t = torch.abs(t)
        return t
    elif val == 'zeros':
        return torch.zeros(data_shape, device=device, dtype=dtype)
    elif val == 'ones':
        return torch.ones(data_shape, device=device, dtype=dtype)
    elif val == 'alternating':
        # initialize the state as alternating pairs of (0,1) (to be used with BlockSkewSymmetric)
        # data_shape is (batch_size, xi_shape)
        assert len(data_shape) == 2, f"xi should be initialized as (batch_size, xi_shape), got xi with {len(data_shape)} dimensions."
        order = data_shape[1] // 2
        batch_size = data_shape[0]
        return torch.tensor([[0., 1.]], device=device, dtype=dtype).repeat(batch_size, order)


def _init_state_and_costate(model: nn.Module, batch_size: int = 1) -> tuple[dict[str, torch.Tensor | dict[str, torch.nn.Parameter]], ...]:
    """Initialize the state and costate dictionaries (keys are 'xi', 'w_xi', 'w_y').
    """

    # creating state and costate
    x = {'xi': model.h_init, 'w': {}}
    p = {'xi': torch.zeros_like(x['xi']), 'w': {}}

    # getting device
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # initialize state and costate for the weights of the state network
    x['w'] = {par_name: par for par_name, par in dict(model.named_parameters()).items() if par.requires_grad}
    p['w'] = {param_name: _init('zeros', param.shape, device, dtype) for param_name, param in x['w'].items()}

    return x, p


def _get_grad(a: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
    """Collects gradients from a list of Tensors.

    Params:
        a: list or dict of Tensors.

    Returns:
        List or dict of references to the 'grad' fields of each component of the input list.
    """
    if isinstance(a, dict):
        g = {_k_a: a[_k_a].grad if a[_k_a].grad is not None else torch.zeros_like(a[_k_a]) for _k_a in a.keys()}
    elif isinstance(a, torch.Tensor):
        g = a.grad if a.grad is not None else torch.zeros_like(a)
    else:
        raise Exception(f'Input to this function should be either list or dict, got {type(a)}.')

    return g


def _apply1(a: torch.Tensor | dict[str, torch.Tensor], op: torch.func) \
        -> torch.Tensor | dict[str, torch.Tensor]:
    """Apply an operation to each element on a list or dict of Tensors.

    Params:
        a: list or dict of Tensors.
        op: operation to be applied to the elements in list_ten (it could be a lambda expression).

    Returns:
        List or dict of Tensors with the result of the operation (same size of each list-argument).
    """
    if isinstance(a, dict):
        oo = {_k_a: op(a[_k_a]) for _k_a in a.keys()}
    elif isinstance(a, torch.Tensor):
        oo = op(a)
    else:
        raise Exception(f'Input to this function should be either list or dict, got {type(a)}.')

    return oo


def _apply2(a: torch.Tensor | dict[str, torch.Tensor], b: torch.Tensor | dict[str, torch.Tensor],
           op: torch.func) -> torch.Tensor | dict[str, torch.Tensor]:
    """Apply an operation involving each pair of elements stored into two lists or dicts of Tensors.

    Params:
        a: first list or dict of Tensors.
        b: second list or dict of Tensors.
        op: operation to be applied to the elements in both the lists (it could be a lambda expression).

    Returns:
        List or dict of Tensors with the result of the operation (same size of each list-argument).
    """
    assert type(a) is type(b), f'Type of the inputs to this function should match, got {type(a)} and {type(b)} instead.'

    if isinstance(a, dict):
        assert set(a.keys()) == set(b.keys()), 'Dictionaries should have the same keys.'
        oo = {_k_a: op(a[_k_a], b[_k_a]) for _k_a in a.keys()}
    elif isinstance(a, torch.Tensor):
        oo = op(a, b)
    else:
        raise Exception(f'Input to this function should be either list or dict, got {type(a)}.')

    return oo


def _copy_inplace(a: torch.Tensor | dict[str, torch.Tensor], b: torch.Tensor | dict[str, torch.Tensor],
                 detach: bool = False) -> None:
    """Copies 'in-place' the values of a list or dict of Tensors (b) into another one (a).

    Params:
        a: list or dict of Tensors.
        b: another list or dict of Tensors, same sizes of the one above.
    """
    assert type(a) is type(b), f'Type of the inputs to this function should match, got {type(a)} and {type(b)} instead.'

    if detach:
        b = _detach(b)
    if isinstance(a, torch.Tensor):
        a.copy_(b)
    elif isinstance(a, dict):
        for _k_b in b.keys():
            a[_k_b].copy_(b[_k_b])
    else:
        raise Exception(f'Inputs to this function should be either lists or dicts, got {type(a)} and {type(b)}.')


def _copy(a: torch.Tensor | dict[str, torch.Tensor], detach: bool = False) \
        -> torch.Tensor | dict[str, torch.Tensor]:
    """Copies the values of a list or dict of Tensors into another one.

    Params:
        a: list or dict of Tensors.

    Returns:
        b: another list or dict of Tensors, same sizes of the one above.
    """
    if detach:
        a = _detach(a)
    if isinstance(a, torch.Tensor):
        b = a.clone()
    elif isinstance(a, dict):
        b = {k: v.clone() for k, v in a.items()}
    else:
        raise Exception(f'Input to this function should be either list or dict, got {type(a)}.')

    return b


def _zero_grad(tensors: torch.Tensor | dict[str, torch.Tensor], set_to_none: bool = False) -> None:
    # todo cambia le docstrings
    """Zeroes the gradient field of a list or dict of Tensors.

    Params:
        a: a list or dict of Tensors with requires_grad activated.
        set_to_none: forces all 'grad' fields to be set to None.
    """
    if isinstance(tensors, dict):
        list_ten = list(tensors.values())
    elif isinstance(tensors, torch.Tensor):
        list_ten = [tensors, ]
    else:
        # todo cambia list con tensor
        raise Exception(f'Input to this function should be either list or dict, got {type(tensors)}.')
    for a in list_ten:
        if a.grad is not None:
            if set_to_none:
                a.grad = None
            else:
                if a.grad.grad_fn is not None:
                    a.grad.detach_()
                else:
                    a.grad.requires_grad_(False)
                    a.grad.zero_()


def _zero(tensors: torch.Tensor | dict[str, torch.Tensor], detach: bool = False) \
        -> torch.Tensor | dict[str, torch.Tensor]:
    """Returns a zeroed copy of a list or dict of tensors.

    Params:
        tensors: a list or dict of Tensors.
    """
    if detach:
        tensors = _detach(tensors)
    if isinstance(tensors, torch.Tensor):
        b = torch.zeros_like(tensors)
    elif isinstance(tensors, dict):
        b = {k: torch.zeros_like(v) for k, v in tensors.items()}
    else:
        raise Exception(f'Input to this function should be either list or dict, got {type(tensors)}.')

    return b


def _zero_inplace(tensors: torch.Tensor | dict[str, torch.Tensor], detach: bool = False) -> None:
    """Zeroes a list or dict of Tensors (inplace).

    Params:
        tensors: a list or dict of Tensors.
    """
    if detach:
        tensors = _detach(tensors)
    if isinstance(tensors, dict):
        for a in tensors.values():
            a.zero_()
    elif isinstance(tensors, torch.Tensor):
        tensors.zero_()
    else:
        raise ValueError('Unsupported type.')


def _detach(tensors: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
    """Detaches a list or dict of Tensors (not-in-place).

    Params:
        a: a list or dict of Tensors.

    Returns:
        A list or dict of detached Tensors.
    """
    if isinstance(tensors, dict):
        oo = dict.fromkeys(tensors)
        for k, a in tensors.items():
            oo[k] = a.detach()
    elif isinstance(tensors, torch.Tensor):
        oo = tensors.detach()
    else:
        raise ValueError('Unsupported type.')
    return oo


class HL:
    def __init__(self, model: nn.Module, *, gamma=1., flip=-1., theta=0.1, beta=1.,
                 reset_neuron_costate=False, reset_weight_costate=False):
        # Initialize the optimizer
        # Model here is your neural network instance, which you will access during optimization
        self.model = model
        self.delta = self.model.delta

        if gamma < 0.0:
            raise ValueError(f"Invalid gamma: {gamma}, should be >= 0.")
        if beta < 0.0:
            raise ValueError(f"Invalid beta value: {beta}, should be >= 0.")
        if 0.0 < theta or theta > 1.0:
            raise ValueError(f"Invalid theta value: {theta}, should be in [0., 1.]")

        # HL parameters
        defaults = dict(gamma=gamma, flip=flip, theta=theta, beta=beta,
                        reset_neuron_costate=reset_neuron_costate, reset_weight_costate=reset_weight_costate)
        for key, value in defaults.items():
            setattr(self, key, value)
        # state and costate initialization
        self.x, self.p = _init_state_and_costate(self.model)

    @torch.no_grad()
    def step(self):
        self.x['xi'] = self.model.h.detach().clone()
        # update the costates
        dp_xi = _get_grad(self.model.h)
        dp_w = _get_grad(self.x['w'])  # dictionary
        _euler_step(self.p['xi'], dp_xi, step_size=-self.delta * self.flip, decay=self.flip * self.theta, in_place=True)
        _euler_step(self.p['w'], dp_w, step_size=-self.delta * self.flip, decay=self.flip * self.theta, in_place=True)

        # update the weights
        _euler_step(self.x['w'], self.p['w'], step_size=-self.delta * self.beta, decay=None, in_place=True)

        # copy back the weights into the network
        for name, param in self.model.named_parameters():
            param.copy_(self.x['w'][name])

    def compute_hamiltonian(self, potential_term: torch.Tensor) -> torch.Tensor:
        """Computes the reduced hamiltonian when provided with the potential term ("""
        return self.gamma * potential_term + torch.dot(self.model.dh.view(-1), self.p['xi'].view(-1)).real

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zeroes the gradients of interest and eventually resets the co-states. This last operation requires us
        to call zero_grad AFTER the backward pass."""
        _zero_grad(self.model.h, set_to_none)
        _zero_grad(self.x['w'], set_to_none)
        if self.reset_neuron_costate:
            _zero_inplace(self.p['xi'], detach=True)
        if self.reset_weight_costate:
            _zero_inplace(self.p['w'], detach=True)
