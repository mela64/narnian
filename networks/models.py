import torch
import random
import numpy as np
import torchvision
from typing import Callable
import torch.nn.functional as F
from networks.cnu.layers import LinearCNU
from networks.cnu.cnus import CNUs


def hard_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=-1., max=1.)


def set_seed(seed: int) -> None:
    if seed >= 0:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(0)


class BasicGenerator(torch.nn.Module):
    """ Basic generator model with linear transformations and a recurrent hidden state """
    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, sigma: Callable = F.tanh,
                 project_every: int = 0, local: bool = False, device: torch.device = torch.device("cpu")):
        super(BasicGenerator, self).__init__()
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim
        
        # Define linear transformation matrices for state update and output mapping
        self.A = torch.nn.Linear(h_dim, h_dim, bias=False, device=device)  # Recurrent weight matrix
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)  # Input-to-hidden mapping
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)  # Hidden-to-output mapping

        # Hidden state initialization
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True)
        self.h_init = self.h.clone()
        self.h_next = torch.empty((1, h_dim), device=device, requires_grad=False)
        self.dh = torch.zeros_like(self.h, device=device, requires_grad=True)
        self.sigma = sigma  # the non-linear activation function

        # Store input dimensions and device
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = 1.  # Discrete time step
        self.local = local  # if True the state update is computed locally in time (i.e., kept out from the graph)
        self.forward_count = 0
        self.project_every = project_every

    @torch.no_grad()
    def adjust_eigs(self):
        """Placeholder for eigenvalue adjustment method"""
        pass

    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.h_init

    @staticmethod
    def handle_inputs(du, u):
        # in the general case DO NOTHING
        # return du, u
        return torch.zeros_like(du), torch.zeros_like(u)

    def forward(self, u, du, first=False):
        """ Forward pass that updates the hidden state and computes the output. """

        # Handle missing inputs
        u = u.flatten(1).to(self.device) if u is not None else torch.zeros((1, self.u_dim), device=self.device)
        du = du.to(self.device) if du is not None else torch.zeros((1, self.du_dim), device=self.device)

        # Reset hidden state if first step
        if first:
            h = self.init_h(torch.cat([du, u], dim=1))
            self.forward_count = 0
        else:
            h = self.h_next
        # track the gradients on h from here on
        h.requires_grad_()

        # check if it's time to project the eigenvalues
        if self.project_every:
            if self.forward_count % self.project_every == 0:
                self.adjust_eigs()

        # handle inputs
        du, u = self.handle_inputs(du, u)

        # Update hidden state based on input and previous hidden state
        h_new = self.A(h) + self.B(torch.cat([du, u], dim=1))

        if self.local:
            # in the local version we keep track in self.h of the old value of the state
            self.h = h
            self.dh = (h_new - self.h) / self.delta  # (h_new - h_old) / delta
        else:
            # in the non-local version we keep track in self.h of the new value of the state
            self.h = h_new
            self.dh = (self.h - h) / self.delta  # (h_new - h_old) / delta
        # self.h.retain_grad()

        # Compute output using a nonlinear activation function
        y = self.C(self.sigma(self.h))

        # store the new state for the next iteration
        self.h_next = h_new.detach()
        self.forward_count += 1

        return y


class _DiagR(torch.nn.Module):
    """ Diagonal matrix-based generator with real-valued transformations """
    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, sigma: Callable = lambda x: x,
                 project_every: int = 0, local: bool = False, device: torch.device = torch.device("cpu")):
        super(_DiagR, self).__init__()
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim

        # Define diagonal transformation and linear layers
        self.diag = torch.nn.Linear(in_features=1, out_features=h_dim, bias=False, device=device, dtype=torch.float32)
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)

        # Hidden state initialization
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True)
        self.h_init = self.h.clone()
        self.h_next = torch.empty((1, h_dim), device=device, requires_grad=False)
        self.dh = torch.zeros_like(self.h, device=device, requires_grad=True)
        self.sigma = sigma  # the non-linear activation function

        # Store input dimensions and device
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = 1.
        self.local = local  # if True the state update is computed locally in time (i.e., kept out from the graph)
        self.forward_count = 0
        self.project_every = project_every

    @torch.no_grad()
    def adjust_eigs(self):
        """ Normalize the diagonal weight matrix by setting signs. """
        self.diag.weight.copy_(torch.sign(self.diag.weight))

    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.h_init

    @staticmethod
    def handle_inputs(du, u):
        # in the general case DO NOTHING
        # return du, u
        return torch.zeros_like(du), torch.zeros_like(u)

    def forward(self, u, du, first=False):
        """ Forward pass with diagonal transformation. """

        # Handle missing inputs
        u = u.flatten(1).to(self.device) if u is not None else torch.zeros((1, self.u_dim), device=self.device)
        du = du.to(self.device) if du is not None else torch.zeros((1, self.du_dim), device=self.device)

        # Reset hidden state if first step
        if first:
            h = self.init_h(torch.cat([du, u], dim=1))
            self.forward_count = 0
        else:
            h = self.h_next
        # track the gradients on h from here on
        h.requires_grad_()

        # check if it's time to project the eigenvalues
        if self.project_every:
            if self.forward_count % self.project_every == 0:
                self.adjust_eigs()

        # handle inputs
        du, u = self.handle_inputs(du, u)

        # Apply diagonal transformation to hidden state
        h_new = self.diag.weight.view(self.diag.out_features) * h + self.B(torch.cat([du, u], dim=1))

        if self.local:
            # in the local version we keep track in self.h of the old value of the state
            self.h = h
            self.dh = (h_new - self.h) / self.delta  # (h_new - h_old) / delta
        else:
            # in the non-local version we keep track in self.h of the new value of the state
            self.h = h_new
            self.dh = (self.h - h) / self.delta  # (h_new - h_old) / delta
        # self.h.retain_grad()

        # Compute output using a nonlinear activation function
        y = self.C(self.sigma(self.h))

        # store the new state for the next iteration
        self.h_next = h_new.detach()
        self.forward_count += 1

        return y


class _DiagC(torch.nn.Module):
    """ Diagonal matrix-based generator with complex-valued transformations """
    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, sigma: Callable = lambda x: x,
                 project_every: int = 0, local: bool = False, device: torch.device = torch.device("cpu")):
        super(_DiagC, self).__init__()
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim

        # Define diagonal transformation with complex numbers
        self.diag = torch.nn.Linear(in_features=1, out_features=h_dim, bias=False, device=device, dtype=torch.cfloat)
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device, dtype=torch.cfloat)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device, dtype=torch.cfloat)

        # Hidden state initialization
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True, dtype=torch.cfloat)
        self.h_init = self.h.clone()
        self.h_next = torch.empty((1, h_dim), device=device, requires_grad=False, dtype=torch.cfloat)
        self.dh = torch.zeros_like(self.h, device=device, requires_grad=True, dtype=torch.cfloat)
        self.sigma = sigma  # the non-linear activation function

        # Store input dimensions and device
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = 1.
        self.local = local  # if True the state update is computed locally in time (i.e., kept out from the graph)
        self.forward_count = 0
        self.project_every = project_every

    @torch.no_grad()
    def adjust_eigs(self):
        """ Normalize the diagonal weight matrix by dividing by its magnitude. """
        self.diag.weight.div_(self.diag.weight.abs())

    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.h_init

    @staticmethod
    def handle_inputs(du, u):
        # in the general case DO NOTHING
        return torch.zeros_like(du), torch.zeros_like(u)
        # return du, u

    def forward(self, u, du, first=False):
        """ Forward pass with complex-valued transformation. """

        # Handle missing inputs
        u = u.flatten(1).to(self.device) if u is not None else torch.zeros((1, self.u_dim), device=self.device, dtype=torch.cfloat)
        du = du.to(self.device) if du is not None else torch.zeros((1, self.du_dim), device=self.device, dtype=torch.cfloat)

        # Reset hidden state if first step
        if first:
            h = self.init_h(torch.cat([du, u], dim=1))
            self.forward_count = 0
        else:
            h = self.h_next
        # track the gradients on h from here on
        h.requires_grad_()

        # check if it's time to project the eigenvalues
        if self.project_every:
            if self.forward_count % self.project_every == 0:
                self.adjust_eigs()

        # handle inputs
        du, u = self.handle_inputs(du, u)

        # Apply complex diagonal transformation
        h_new = self.diag.weight.view(self.diag.out_features) * h + self.B(torch.cat([du, u], dim=1))

        if self.local:
            # in the local version we keep track in self.h of the old value of the state
            self.h = h
            self.dh = (h_new - self.h) / self.delta  # (h_new - h_old) / delta
        else:
            # in the non-local version we keep track in self.h of the new value of the state
            self.h = h_new
            self.dh = (self.h - h) / self.delta  # (h_new - h_old) / delta
        # self.h.retain_grad()

        # Compute output using a nonlinear activation function
        y = self.C(self.sigma(self.h))

        # store the new state for the next iteration
        self.h_next = h_new.detach()
        self.forward_count += 1

        return y.real


class _CTE(torch.nn.Module):
    """Antisymmetric Matrix Exponential Generator implementing continuous-time dynamics.

    Uses antisymmetric weight matrix with matrix exponential for stable hidden state evolution.

    Args:
        u_shape: Input shape (tuple of integers)
        d_dim: Input descriptor dimension
        y_dim: Output dimension
        h_dim: Hidden state dimension
        delta: Time step for discrete approximation
        device: Computation device (CPU/GPU)
    """

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, delta: float,
                 sigma: Callable = lambda x: x, project_every: int = 0, local: bool = False,
                 device: torch.device = torch.device("cpu"), cnu_memories: int = 0):
        super(_CTE, self).__init__()
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim

        # Antisymmetric weight matrix (W - W^T)
        if cnu_memories <= 0:
            self.W = torch.nn.Linear(h_dim, h_dim, bias=False, device=device)
            self.I = torch.eye(h_dim, requires_grad=False, device=device)  # Identity matrix
            # Input projection matrix
            self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
            # Output projection matrix
            self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)
        else:
            self.W = LinearCNU(h_dim, h_dim, bias=False, device=device, key_size=u_dim + du_dim,
                               delta=1, beta_k=delta, scramble=False, key_mem_units=cnu_memories, shared_keys=True)
            self.I = torch.eye(h_dim, requires_grad=False, device=device)  # Identity matrix
            # Input projection matrix
            self.B = LinearCNU(u_dim + du_dim, h_dim, bias=False, device=device, key_size=u_dim + du_dim,
                               delta=1, beta_k=delta, scramble=False, key_mem_units=cnu_memories, shared_keys=True)
            # Output projection matrix
            self.C = LinearCNU(h_dim, y_dim, bias=False, device=device, key_size=u_dim + du_dim,
                               delta=1, beta_k=delta, scramble=False, key_mem_units=cnu_memories, shared_keys=True)

        # Hidden state initialization
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True)
        self.h_init = self.h.clone()
        self.h_next = torch.empty((1, h_dim), device=device, requires_grad=False)
        self.dh = torch.zeros_like(self.h, device=device, requires_grad=True)
        self.sigma = sigma  # the non-linear activation function

        # System parameters
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = delta
        self.local = local
        self.forward_count = 0
        self.project_every = project_every

    @torch.no_grad()
    def adjust_eigs(self):
        """Placeholder for eigenvalue adjustment method"""
        pass

    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.h_init

    @staticmethod
    def handle_inputs(du, u):
        # in the general case DO NOTHING
        return du, u

    def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = False) -> torch.Tensor:
        """Forward pass through the system dynamics.

        Args:
            u: Input tensor of shape (batch_size, u_dim)
            du: Input descriptor tensor of shape (batch_size, du_dim)
            first: Flag indicating first step (resets hidden state)

        Returns:
            y: Output tensor of shape (batch_size, y_dim)
        """
        # Handle missing inputs
        u = u.flatten(1).to(self.device) if u is not None else torch.zeros((1, self.u_dim), device=self.device)
        du = du.to(self.device) if du is not None else torch.zeros((1, self.du_dim), device=self.device)

        # Reset hidden state if first step
        if first:
            h = self.init_h(torch.cat([du, u], dim=1))
            self.forward_count = 0
        else:
            h = self.h_next
        # track the gradients on h from here on
        h.requires_grad_()

        # check if it's time to project the eigenvalues
        if self.project_every:
            if self.forward_count % self.project_every == 0:
                self.adjust_eigs()

        if not isinstance(self.W, LinearCNU):
            weight_W = self.W.weight
            B = self.B
            C = self.C
        else:
            udu = torch.cat([du, u], dim=1)
            weight_W = self.W.compute_weights(udu).view(self.W.out_features, self.W.in_features)
            weight_B = self.B.compute_weights(udu).view(self.B.out_features, self.B.in_features)
            weight_C = self.C.compute_weights(udu).view(self.C.out_features, self.C.in_features)
            B = lambda x: torch.nn.functional.linear(x, weight_B)
            C = lambda x: torch.nn.functional.linear(x, weight_C)

        # handle inputs
        du, u = self.handle_inputs(du, u)

        # Antisymmetric matrix construction
        A = 0.5 * (weight_W - weight_W.t())
        A_expm = torch.linalg.matrix_exp(A * self.delta)  # Matrix exponential
        rec = F.linear(h, A_expm, self.W.bias)  # Recurrent component

        # Input processing component
        A_inv = torch.linalg.inv(A)
        inp = A_inv @ (A_expm - self.I) @ B(torch.cat([du, u], dim=1)).unsqueeze(-1)

        # Handle locality
        h_new = rec + inp.squeeze(-1)   # updated hidden state
        if self.local:
            # in the local version we keep track in self.h of the old value of the state
            self.h = h
            self.dh = (h_new - self.h) / self.delta     # (h_new - h_old) / delta
        else:
            # in the non-local version we keep track in self.h of the new value of the state
            self.h = h_new
            self.dh = (self.h - h) / self.delta     # (h_new - h_old) / delta
        # self.h.retain_grad()

        # Compute output using a nonlinear activation function
        y = C(self.sigma(self.h))

        # store the new state for the next iteration
        self.h_next = h_new.detach()

        return y


class _CTB(torch.nn.Module):
    """Block Antisymmetric Generator using 2x2 parameterized rotation blocks.

    Implements structured antisymmetric dynamics through learnable rotational frequencies.

    Args:
        u_shape: Input shape (tuple of integers)
        d_dim: Input descriptor dimension
        y_dim: Output dimension
        h_dim: Hidden state dimension
        delta: Time step for discrete approximation
        alpha: Dissipation added on the diagonal (also controls the eigenvalue projections method)
        device: Computation device (CPU/GPU)
    """

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, delta: float = None,
                 alpha: float = 0., sigma: Callable = lambda x: x, project_every: int = 0, local: bool = False,
                 device: torch.device = torch.device("cpu")):
        super(_CTB, self).__init__()
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim

        assert h_dim % 2 == 0, "Hidden dimension must be even for 2x2 blocks"
        self.order = h_dim // 2  # Number of 2x2 blocks

        # Learnable rotational frequencies
        self.omega = torch.nn.Parameter(torch.empty(self.order, device=device))
        self.register_buffer('ones', torch.ones(self.order, requires_grad=False, device=device))

        # Projection matrices
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)

        # Damping configuration
        if alpha > 0.:
            # in this case we want to add the feedback parameter alpha and use it to move eigenvalues on the unit circle
            self.project_method = 'const'
            self.register_buffer('alpha', torch.full_like(self.omega.data, alpha, device=device))
        elif alpha == 0.:
            # this is the case in which we want to divide by the modulus
            self.project_method = 'modulus'
            self.register_buffer('alpha', torch.zeros_like(self.omega.data, device=device))
        elif alpha == -1.:
            self.project_method = 'alpha'
            self.register_buffer('alpha', torch.zeros_like(self.omega.data, device=device))

        # Hidden state initialization
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True)
        self.h_init = self.h.clone()
        self.h_next = torch.empty((1, h_dim), device=device, requires_grad=False)
        self.dh = torch.zeros_like(self.h, device=device, requires_grad=True)
        self.sigma = sigma  # the non-linear activation function

        # System parameters
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = delta
        self.local = local  # if True the state update is computed locally in time (i.e., kept out from the graph)
        self.reset_parameters()
        self.forward_count = 0
        self.project_every = project_every

    def reset_parameters(self) -> None:
        """Initialize rotational frequencies with uniform distribution"""
        torch.nn.init.uniform_(self.omega)

    @torch.no_grad()
    def adjust_eigs(self):
        """Adjust eigenvalues to maintain stability"""
        with torch.no_grad():
            if self.project_method == 'alpha':
                # Compute damping to maintain eigenvalues on unit circle
                self.alpha.copy_((1. - torch.sqrt(1. - (self.delta * self.omega) ** 2) / self.delta))
            elif self.project_method == 'modulus':
                # Normalize by modulus for unit circle stability
                module = torch.sqrt(self.ones ** 2 + (self.delta * self.omega) ** 2)
                self.omega.div_(module)
                self.ones.div_(module)

    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.h_init

    @staticmethod
    def handle_inputs(du, u):
        # in the general case DO NOTHING
        return torch.zeros_like(du), torch.zeros_like(u)
        # return du, u

    def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = False) -> torch.Tensor:
        """Forward pass through block-structured dynamics"""
        # Handle missing inputs
        u = u.flatten(1).to(self.device) if u is not None else torch.zeros((1, self.u_dim), device=self.device)
        du = du.to(self.device) if du is not None else torch.zeros((1, self.du_dim), device=self.device)

        # Reset hidden state if first step
        if first:
            h = self.init_h(torch.cat([du, u], dim=1))
            self.forward_count = 0
        else:
            h = self.h_next
        # track the gradients on h from here on
        h.requires_grad_()
        h_pair = h.view(-1, self.order, 2)  # Reshape to (batch, blocks, 2)

        # check if it's time to project the eigenvalues
        if self.project_every:
            if self.forward_count % self.project_every == 0:
                self.adjust_eigs()

        # handle inputs
        du, u = self.handle_inputs(du, u)

        # Block-wise rotation with damping
        h1 = (self.ones - self.delta * self.alpha) * h_pair[..., 0] + self.delta * self.omega * h_pair[..., 1]
        h2 = -self.delta * self.omega * h_pair[..., 0] + (self.ones - self.delta * self.alpha) * h_pair[..., 1]

        # Recurrent and input components
        rec = torch.stack([h1, h2], dim=-1).flatten(start_dim=1)
        inp = self.delta * self.B(torch.cat([du, u], dim=1))

        # Handle locality
        h_new = rec + inp  # updated hidden state
        if self.local:
            # in the local version we keep track in self.h of the old value of the state
            self.h = h
            self.dh = (h_new - self.h) / self.delta  # (h_new - h_old) / delta
        else:
            # in the non-local version we keep track in self.h of the new value of the state
            self.h = h_new
            self.dh = (self.h - h) / self.delta  # (h_new - h_old) / delta
        # self.h.retain_grad()

        # Compute output using a nonlinear activation function
        y = self.C(self.sigma(self.h))

        # store the new state for the next iteration
        self.h_next = h_new.detach()
        self.forward_count += 1

        return y


class _CTBE(torch.nn.Module):
    """Antisymmetric Generator with Exact Matrix Exponential Blocks.

    Implements precise rotational dynamics using trigonometric parameterization.

    Args:
        u_shape: Input shape (tuple of integers)
        d_dim: Input descriptor dimension
        y_dim: Output dimension
        h_dim: Hidden state dimension
        delta: Time step for discrete approximation
        device: Computation device (CPU/GPU)
    """

    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, delta: float,
                 sigma: Callable = lambda x: x, project_every: int = 0, local: bool = False,
                 device: torch.device = torch.device("cpu"), cnu_memories: int = 0):
        super(_CTBE, self).__init__()
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim

        assert h_dim % 2 == 0, "Hidden dimension must be even for 2x2 blocks"
        self.order = h_dim // 2

        # Learnable rotational frequencies
        if cnu_memories <= 0:
            self.omega = torch.nn.Parameter(torch.empty(self.order, device=device))
            self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
            self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)
        else:
            self.omega = CNUs(q=1, d=u_dim + du_dim, u=self.order,
                              delta=1, beta_k=delta, scramble=False, m=cnu_memories)
            self.B = LinearCNU(u_dim + du_dim, h_dim, bias=False, device=device, key_size=u_dim + du_dim,
                               delta=1, beta_k=delta, scramble=False, key_mem_units=cnu_memories, shared_keys=True)
            self.C = LinearCNU(h_dim, y_dim, bias=False, device=device, key_size=u_dim + du_dim,
                               delta=1, beta_k=delta, scramble=False, key_mem_units=cnu_memories, shared_keys=True)

        # Hidden state initialization
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True)
        self.h_init = self.h.clone()
        self.h_next = torch.empty((1, h_dim), device=device, requires_grad=False)
        self.dh = torch.zeros_like(self.h, device=device, requires_grad=True)
        self.sigma = sigma  # the non-linear activation function

        # System parameters
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = delta
        self.local = local  # if True the state update is computed locally in time (i.e., kept out from the graph)
        self.reset_parameters()
        self.forward_count = 0
        self.project_every = project_every

    def reset_parameters(self) -> None:
        """Initialize rotational frequencies"""
        if isinstance(self.omega, torch.nn.Parameter):
            torch.nn.init.uniform_(self.omega)
        else:
            torch.nn.init.uniform_(self.omega.M)

    @torch.no_grad()
    def adjust_eigs(self):
        """Placeholder for eigenvalue adjustment"""
        pass

    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.h_init

    @staticmethod
    def handle_inputs(du, u):
        # in the general case DO NOTHING
        return torch.zeros_like(du), torch.zeros_like(u)
        # return du, u

    def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = False) -> torch.Tensor:
        """Exact matrix exponential forward pass"""
        # Handle missing inputs
        u = u.flatten(1).to(self.device) if u is not None else torch.zeros((1, self.u_dim), device=self.device)
        du = du.to(self.device) if du is not None else torch.zeros((1, self.du_dim), device=self.device)

        # Reset hidden state if first step
        if first:
            h = self.init_h(torch.cat([du, u], dim=1))
            self.forward_count = 0
        else:
            h = self.h_next
        # track the gradients on h from here on
        h.requires_grad_()
        h_pair = h.view(-1, self.order, 2)

        # check if it's time to project the eigenvalues
        if self.project_every:
            if self.forward_count % self.project_every == 0:
                self.adjust_eigs()

        if isinstance(self.omega, torch.nn.Parameter):
            omega = self.omega
            B = self.B
            C = self.C
        else:
            udu = torch.cat([du, u], dim=1)
            omega = self.omega.compute_weights(udu).view(-1)
            weight_B = self.B.compute_weights(udu).view(self.B.out_features, self.B.in_features)
            weight_C = self.C.compute_weights(udu).view(self.C.out_features, self.C.in_features)
            B = lambda x: torch.nn.functional.linear(x, weight_B)
            C = lambda x: torch.nn.functional.linear(x, weight_C)

        # handle inputs
        du, u = self.handle_inputs(du, u)
        udu = torch.cat([du, u], dim=1)

        # Trigonometric terms for exact rotation
        cos_t = torch.cos(omega * self.delta)
        sin_t = torch.sin(omega * self.delta)

        # Rotational update
        h1 = cos_t * h_pair[..., 0] + sin_t * h_pair[..., 1]
        h2 = -sin_t * h_pair[..., 0] + cos_t * h_pair[..., 1]
        rec = torch.stack([h1, h2], dim=-1).flatten(start_dim=1)

        # Input processing
        u_hat = B(udu).view(-1, self.order, 2)
        inp1 = (sin_t * u_hat[..., 0] - (cos_t - 1) * u_hat[..., 1]) / omega
        inp2 = ((cos_t - 1) * u_hat[..., 0] + sin_t * u_hat[..., 1]) / omega
        inp = torch.stack([inp1, inp2], dim=-1).flatten(start_dim=1)

        # Handle locality
        h_new = rec + inp  # updated hidden state
        if self.local:
            # in the local version we keep track in self.h of the old value of the state
            self.h = h
            self.dh = (h_new - self.h) / self.delta  # (h_new - h_old) / delta
        else:
            # in the non-local version we keep track in self.h of the new value of the state
            self.h = h_new
            self.dh = (self.h - h) / self.delta  # (h_new - h_old) / delta
        # self.h.retain_grad()

        # Compute output using a nonlinear activation function
        y = C(self.sigma(self.h))

        # store the new state for the next iteration
        self.h_next = h_new.detach()
        self.forward_count += 1

        return y


class BasicPredictor(torch.nn.Module):
    """Simple Predictive Network with Tanh Nonlinearity

    Args:
        y_dim: Input observation dimension
        d_dim: Prediction dimension
        h_dim: Hidden state dimension
    """

    def __init__(self, y_dim: int, d_dim: int, h_dim: int, device: torch.device = torch.device("cpu")):
        super(BasicPredictor, self).__init__()
        self.device = device

        # System matrices
        self.A = torch.nn.Linear(h_dim, h_dim, bias=False, device=device)
        self.B = torch.nn.Linear(y_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, d_dim, bias=False, device=device)

        # Hidden state
        self.register_buffer("h", torch.randn(1, h_dim, device=device))
        self.register_buffer("h_init", self.h.clone())
        self.local = False  # if True the state update is computed locally in time (i.e., kept out from the graph)

    def forward(self, y: torch.Tensor, first: bool = False) -> torch.Tensor:
        """Simple nonlinear prediction step"""
        y = y.to(self.device)

        if first:
            self.h = self.h_init

        # State update
        h = self.A(self.h) + self.B(y)
        d = self.C(torch.tanh(h))  # Nonlinear projection

        # Detach state for next iteration
        self.h.data = h.detach()

        return d


class BasicImagePredictor(torch.nn.Module):

    def __init__(self, d_dim: int, device: torch.device = torch.device("cpu")):
        super(BasicImagePredictor, self).__init__()
        self.device = device

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.CenterCrop(32),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 3 * 3, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, d_dim),
            torch.nn.Sigmoid()
        ).to(self.device)

    def forward(self, y, first=False):
        return self.net(self.transforms(y).to(self.device))


class BasicImagePredictorCNU(torch.nn.Module):

    def __init__(self, d_dim: int, mem_units: int, device: torch.device = torch.device("cpu"),
                 delta: int = 1, scramble: bool = False):
        super(BasicImagePredictorCNU, self).__init__()
        self.device = device

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.CenterCrop(32),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=3, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 3 * 3, 2048),
            torch.nn.ReLU(inplace=True),
            LinearCNU(2048, d_dim, key_mem_units=mem_units, delta=delta, scramble=scramble),
            torch.nn.Sigmoid()
        ).to(self.device)

    def forward(self, y, first=False):
        return self.net(self.transforms(y).to(self.device))


class BasicTokenGenerator(torch.nn.Module):

    def __init__(self, num_emb: int, emb_dim: int, d_dim: int, y_dim: int, h_dim: int,
                 device: torch.device = torch.device("cpu"), seed: int = -1):
        super(BasicTokenGenerator, self).__init__()
        self.device = device
        set_seed(seed)

        u_dim = emb_dim
        du_dim = d_dim
        self.embeddings = torch.nn.Embedding(num_emb, emb_dim)

        self.A = torch.nn.Linear(h_dim, h_dim, bias=False, device=device)
        self.B = lambda x: 0.  # killing every input
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)
        self.h_init = torch.randn((1, h_dim), device=device)
        self.u_init = torch.zeros((1, u_dim), device=device)
        self.h = None
        self.u_dim = u_dim
        self.du_dim = du_dim

    def forward(self, u, du, first=False):
        if first:
            h = self.h_init
        else:
            h = self.h.detach()
        if u is None:
            u = torch.zeros((1, self.u_dim), dtype=torch.float32, device=self.device)
        else:
            u = self.embeddings(u.to(self.device))  # added this
        if du is None:
            du = torch.zeros((1, self.du_dim), dtype=torch.float32, device=self.device)
        else:
            du = du.to(self.device)

        self.h = torch.tanh(self.A(h) + self.B(torch.cat([du, u], dim=1)))
        y = self.C(self.h)
        return y


class BasicTokenGeneratorCTE(torch.nn.Module):

    def __init__(self, num_emb: int, emb_dim: int, d_dim: int, y_dim: int, h_dim: int,
                 device: torch.device = torch.device("cpu"), seed: int = -1):
        super(BasicTokenGeneratorCTE, self).__init__()
        self.device = device
        set_seed(seed)

        self.net = AntisymmetricExpGenerator((emb_dim,), d_dim, y_dim, h_dim,
                                             delta=1.0, local=False, device=device)
        self.embeddings = torch.nn.Embedding(num_emb, emb_dim)

    def forward(self, u, du, first=False):
        if u is not None:
            u = self.embeddings(u.to(self.device))
        y = self.net(u, du, first=first)
        return y


class BasicTokenGeneratorLM(torch.nn.Module):

    def __init__(self, num_emb: int, emb_dim: int, d_dim: int, y_dim: int, h_dim: int,
                 device: torch.device = torch.device("cpu"), seed: int = -1):
        super(BasicTokenGeneratorLM, self).__init__()
        self.device = device
        set_seed(seed)

        u_dim = emb_dim
        du_dim = d_dim
        self.embeddings = torch.nn.Embedding(num_emb, emb_dim)

        self.A = torch.nn.Linear(h_dim, h_dim, bias=False, device=device)
        self.B = torch.nn.Linear(u_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)
        self.h_init = torch.randn((1, h_dim), device=device)
        self.u_init = torch.zeros((1, u_dim), device=device)
        self.h = None
        self.y = None
        self.u_dim = u_dim
        self.du_dim = du_dim

    def forward(self, u, du, first=False):
        if first:
            h = self.h_init
            u = self.u_init
        else:
            h = self.h.detach()
            u = self.embeddings((torch.argmax(self.y.detach(), dim=1) if self.y.shape[1] > 1
                                 else self.y.squeeze(1).detach()).to(self.device))

        self.h = torch.tanh(self.A(h) + self.B(u))
        self.y = self.C(self.h)
        return self.y


class BasicTokenPredictor(torch.nn.Module):

    def __init__(self, num_emb: int, emb_dim: int, d_dim: int,  h_dim: int,
                 device: torch.device = torch.device("cpu"), seed: int = -1):
        super(BasicTokenPredictor, self).__init__()
        self.device = device
        set_seed(seed)

        y_dim = emb_dim
        self.embeddings = torch.nn.Embedding(num_emb, emb_dim, device=self.device)

        self.A = torch.nn.Linear(h_dim, h_dim, bias=False, device=self.device)
        self.B = torch.nn.Linear(y_dim, h_dim, bias=False, device=self.device)
        self.C = torch.nn.Linear(h_dim, d_dim, bias=False, device=self.device)
        self.register_buffer("h", torch.randn((1, h_dim), device=self.device))
        self.register_buffer("h_init", self.h.clone())
        self.local = False  # if True the state update is computed locally in time (i.e., kept out from the graph)

    def forward(self, y, first=False):
        if first:
            self.h.data = self.h_init
        y = self.embeddings(y.to(self.device))  # added this
        h = self.A(self.h) + self.B(y)
        d = self.C(torch.tanh(h))
        self.h.data = h.detach()
        return d
