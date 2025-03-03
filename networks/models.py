import torch
import torchvision
import torch.nn.functional as F


class BasicGenerator(torch.nn.Module):
    """ Basic generator model with linear transformations and a recurrent hidden state """
    def __init__(self, u_shape: tuple[int], d_dim: int, y_dim: int, h_dim: int, device: torch.device = torch.device("cpu")):
        super(BasicGenerator, self).__init__()
        u_shape = torch.Size(u_shape)
        u_dim = u_shape.numel()
        du_dim = d_dim
        
        # Define linear transformation matrices for state update and output mapping
        self.A = torch.nn.Linear(h_dim, h_dim, bias=False, device=device)  # Recurrent weight matrix
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)  # Input-to-hidden mapping
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)  # Hidden-to-output mapping

        # Initialize hidden state
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True)  # Trainable hidden state
        self.h_init = self.h.clone()  # Store initial hidden state
        self.dh = torch.zeros_like(self.h)  # Store hidden state derivative

        # Store input dimensions and device
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = 1.  # Discrete time step

    @torch.no_grad()
    def adjust_eigs(self, delta=0.01):
        """ Placeholder for adjusting eigenvalues, not necessary in this case. """
        pass

    def forward(self, u, du, first=False):
        """ Forward pass that updates the hidden state and computes the output. """

        # Handle None inputs by initializing to zero tensors
        if u is None:
            u = torch.zeros((1, self.u_dim), dtype=torch.float32, device=self.device)
        else:
            u = u.flatten(1)
        if du is None:
            du = torch.zeros((1, self.du_dim), dtype=torch.float32, device=self.device)

        h = self.h.detach()  # Detach previous hidden state from computation graph
        if first:
            h = self.h_init  # Reset hidden state if we want to do so at the first step

        # Update hidden state based on input and previous hidden state
        self.h = self.A(h) + self.B(torch.cat([du, u], dim=1))

        # Compute output using a nonlinear activation function
        y = self.C(torch.tanh(self.h))

        # Compute hidden state derivative
        self.dh = (self.h - h) / self.delta

        # Retain gradient of the hidden state
        self.h.retain_grad()

        return y


class DiagReal(torch.nn.Module):
    """ Diagonal matrix-based generator with real-valued transformations """
    def __init__(self, u_dim: int, du_dim: int, y_dim: int, h_dim: int, device: torch.device = torch.device("cpu")):
        super(DiagReal, self).__init__()

        # Define diagonal transformation and linear layers
        self.diag = torch.nn.Linear(in_features=1, out_features=h_dim, bias=False, device=device, dtype=torch.float32)
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)

        # Initialize hidden state
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True)
        self.h_init = self.h.clone()
        self.dh = torch.zeros_like(self.h)

        # Store input dimensions and device
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = 1.

    @torch.no_grad()
    def adjust_eigs(self, delta=0.01):
        """ Normalize the diagonal weight matrix by setting signs. """
        self.diag.weight.copy_(torch.sign(self.diag.weight))

    def forward(self, u, du, first=False):
        """ Forward pass with diagonal transformation. """

        # Handle None inputs
        if u is None:
            u = torch.zeros((1, self.u_dim), dtype=torch.float32, device=self.device)
        if du is None:
            du = torch.zeros((1, self.du_dim), dtype=torch.float32, device=self.device)

        h = self.h.detach()
        if first:
            h = self.h_init

        # Apply diagonal transformation to hidden state
        self.h = self.diag.weight.view(self.diag.out_features) * h + self.B(torch.cat([du, u], dim=1))

        # Compute output
        y = self.C(torch.tanh(self.h))

        # Compute hidden state derivative
        self.dh = (self.h - h) / self.delta

        self.h.retain_grad()
        return y


class DiagCompl(torch.nn.Module):
    """ Diagonal matrix-based generator with complex-valued transformations """
    def __init__(self, u_dim: int, du_dim: int, y_dim: int, h_dim: int, device: torch.device = torch.device("cpu")):
        super(DiagCompl, self).__init__()

        # Define diagonal transformation with complex numbers
        self.diag = torch.nn.Linear(in_features=1, out_features=h_dim, bias=False, device=device, dtype=torch.cfloat)
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device, dtype=torch.cfloat)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device, dtype=torch.cfloat)

        # Initialize hidden state with complex values
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True, dtype=torch.cfloat)
        self.h_init = self.h.clone()
        self.dh = torch.zeros_like(self.h)

        # Store input dimensions and device
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = 1.

    @torch.no_grad()
    def adjust_eigs(self, delta=0.01):
        """ Normalize the diagonal weight matrix by dividing by its magnitude. """
        self.diag.weight.div_(self.diag.weight.abs())

    def forward(self, u, du, first=False):
        """ Forward pass with complex-valued transformation. """

        # Convert input to complex if needed
        if u is None:
            u = torch.zeros((1, self.u_dim), dtype=torch.cfloat, device=self.device)
        else:
            u.to(dtype=torch.cfloat)

        if du is None:
            du = torch.zeros((1, self.du_dim), dtype=torch.cfloat, device=self.device)
        else:
            du.to(dtype=torch.cfloat)

        h = self.h.detach()
        if first:
            h = self.h_init

        # Apply complex diagonal transformation
        self.h = self.diag.weight.view(self.diag.out_features) * h + self.B(torch.cat([du, u], dim=1))

        # Compute real-valued output
        y = self.C(torch.tanh(self.h)).real

        # Compute hidden state derivative
        self.dh = (self.h - h) / self.delta

        self.h.retain_grad()
        return y


class AntisymmetricExpGenerator(torch.nn.Module):
    """Antisymmetric Matrix Exponential Generator implementing continuous-time dynamics.

    Uses antisymmetric weight matrix with matrix exponential for stable hidden state evolution.

    Args:
        u_dim: Input dimension
        du_dim: Input descriptor dimension
        y_dim: Output dimension
        h_dim: Hidden state dimension
        delta: Time step for discrete approximation
        device: Computation device (CPU/GPU)
    """

    def __init__(self, u_dim: int, du_dim: int, y_dim: int, h_dim: int, delta: float,
                 device: torch.device = torch.device("cpu")):
        super(AntisymmetricExpGenerator, self).__init__()
        # Antisymmetric weight matrix (W - W^T)
        self.W = torch.nn.Linear(h_dim, h_dim, bias=False, device=device)
        self.I = torch.eye(h_dim, requires_grad=False, device=device)  # Identity matrix
        # Input projection matrix
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        # Output projection matrix
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)

        # Hidden state initialization
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True)
        self.h_init = self.h.clone()  # save the initial state
        self.dh = torch.zeros_like(self.h)  # Hidden state derivative

        # System parameters
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = delta

    @torch.no_grad()
    def adjust_eigs(self, delta=0.01):
        """Placeholder for eigenvalue adjustment method"""
        pass

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
        u = u if u is not None else torch.zeros((1, self.u_dim), device=self.device)
        du = du if du is not None else torch.zeros((1, self.du_dim), device=self.device)

        # Reset hidden state if first step
        h = self.h_init if first else self.h.detach()

        # Antisymmetric matrix construction
        A = 0.5 * (self.W.weight - self.W.weight.t())
        A_expm = torch.linalg.matrix_exp(A * self.delta)  # Matrix exponential

        # Recurrent component
        rec = F.linear(h, A_expm, self.W.bias)

        # Input processing component
        A_inv = torch.linalg.inv(A)
        inp = A_inv @ (A_expm - self.I) @ self.B(torch.cat([du, u], dim=1)).unsqueeze(-1)

        # Update hidden state
        self.h = rec + inp.squeeze(-1)
        y = self.C(self.h)  # Linear output projection

        # Compute hidden state derivative
        self.dh = (self.h - h) / self.delta
        self.h.retain_grad()  # Preserve gradient for custom backward pass

        return y


class BlockAntisymmetricGenerator(torch.nn.Module):
    """Block Antisymmetric Generator using 2x2 parameterized rotation blocks.

    Implements structured antisymmetric dynamics through learnable rotational frequencies.

    Args:
        u_dim: Input dimension
        du_dim: Input descriptor dimension
        y_dim: Output dimension
        h_dim: Hidden state dimension
        delta: Time step for discrete approximation
        alpha: Dissipation added on the diagonal (also controls the eigenvalue projections method)
        device: Computation device (CPU/GPU)
    """

    def __init__(self, u_dim: int, du_dim: int, y_dim: int, h_dim: int, delta: float = None,
                 alpha: float = 0., device: torch.device = torch.device("cpu")):
        super(BlockAntisymmetricGenerator, self).__init__()
        assert h_dim % 2 == 0, "Hidden dimension must be even for 2x2 blocks"
        self.order = h_dim // 2  # Number of 2x2 blocks

        # Learnable rotational frequencies
        self.omega = torch.nn.Parameter(torch.empty(self.order))
        self.register_buffer('ones', torch.ones(self.order, requires_grad=False))

        # Projection matrices
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)

        # Damping configuration
        if alpha > 0.:
            # in this case we want to add the feedback parameter alpha and use it to move eigenvalues on the unit circle
            self.project_method = 'const'
            self.register_buffer('alpha', torch.full_like(self.omega, alpha))
        elif alpha == 0.:
            # this is the case in which we want to divide by the modulus
            self.project_method = 'modulus'
            self.register_buffer('alpha', torch.zeros_like(self.omega))
        elif alpha == -1.:
            self.project_method = 'alpha'
            self.register_buffer('alpha', torch.zeros_like(self.omega))

        # State initialization
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True)
        self.h_init = self.h.clone()
        self.dh = torch.zeros_like(self.h)

        # System parameters
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = delta
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize rotational frequencies with uniform distribution"""
        torch.nn.init.uniform_(self.omega)

    @torch.no_grad()
    def adjust_eigs(self, delta):
        """Adjust eigenvalues to maintain stability"""
        with torch.no_grad():
            if self.project_method == 'alpha':
                # Compute damping to maintain eigenvalues on unit circle
                self.alpha.copy_((1. - torch.sqrt(1. - (delta * self.omega) ** 2) / delta))
            elif self.project_method == 'modulus':
                # Normalize by modulus for unit circle stability
                module = torch.sqrt(self.ones ** 2 + (self.delta * self.omega) ** 2)
                self.omega.div_(module)
                self.ones.div_(module)

    def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = False) -> torch.Tensor:
        """Forward pass through block-structured dynamics"""
        # Input handling
        u = u if u is not None else torch.zeros((1, self.u_dim), device=self.device)
        du = du if du is not None else torch.zeros((1, self.du_dim), device=self.device)

        # State management
        h = self.h_init if first else self.h.detach()
        h_pair = h.view(-1, self.order, 2)  # Reshape to (batch, blocks, 2)

        # Block-wise rotation with damping
        h1 = (self.ones - self.delta * self.alpha) * h_pair[..., 0] + self.delta * self.omega * h_pair[..., 1]
        h2 = -self.delta * self.omega * h_pair[..., 0] + (self.ones - self.delta * self.alpha) * h_pair[..., 1]

        # Recurrent and input components
        rec = torch.stack([h1, h2], dim=-1).flatten(start_dim=1)
        inp = self.delta * self.B(torch.cat([du, u], dim=1))

        # State update
        self.h = rec + inp
        y = self.C(self.h)

        # Derivative calculation
        self.dh = (self.h - h) / self.delta
        self.h.retain_grad()

        return y


class BlockAntisymmetricExpGenerator(torch.nn.Module):
    """Antisymmetric Generator with Exact Matrix Exponential Blocks.

    Implements precise rotational dynamics using trigonometric parameterization.

    Args:
        u_dim: Input dimension
        du_dim: Input descriptor dimension
        y_dim: Output dimension
        h_dim: Hidden state dimension
        delta: Time step for discrete approximation
        device: Computation device (CPU/GPU)
    """

    def __init__(self, u_dim: int, du_dim: int, y_dim: int, h_dim: int, delta: float,
                 device: torch.device = torch.device("cpu")):
        super(BlockAntisymmetricExpGenerator, self).__init__()
        assert h_dim % 2 == 0, "Hidden dimension must be even for 2x2 blocks"
        self.order = h_dim // 2

        # Learnable rotational frequencies
        self.omega = torch.nn.Parameter(torch.empty(self.order))
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)

        # State initialization
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True)
        self.h_init = self.h.clone()
        self.dh = torch.zeros_like(self.h)

        # System parameters
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = delta
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize rotational frequencies"""
        torch.nn.init.uniform_(self.omega)

    @torch.no_grad()
    def adjust_eigs(self, delta=0.01):
        """Placeholder for eigenvalue adjustment"""
        pass

    def forward(self, u: torch.Tensor, du: torch.Tensor, first: bool = False) -> torch.Tensor:
        """Exact matrix exponential forward pass"""
        # Input handling
        u = u if u is not None else torch.zeros((1, self.u_dim), device=self.device)
        du = du if du is not None else torch.zeros((1, self.du_dim), device=self.device)

        # State management
        h = self.h_init if first else self.h.detach()
        h_pair = h.view(-1, self.order, 2)

        # Trigonometric terms for exact rotation
        cos_t = torch.cos(self.omega * self.delta)
        sin_t = torch.sin(self.omega * self.delta)

        # Rotational update
        h1 = cos_t * h_pair[..., 0] + sin_t * h_pair[..., 1]
        h2 = -sin_t * h_pair[..., 0] + cos_t * h_pair[..., 1]
        rec = torch.stack([h1, h2], dim=-1).flatten(start_dim=1)

        # Input processing
        u_hat = self.B(torch.cat([u, du], 1)).view(-1, self.order, 2)
        inp1 = (sin_t * u_hat[..., 0] - (cos_t - 1) * u_hat[..., 1]) / self.omega
        inp2 = ((cos_t - 1) * u_hat[..., 0] + sin_t * u_hat[..., 1]) / self.omega
        inp = torch.stack([inp1, inp2], dim=-1).flatten(start_dim=1)

        # State update
        self.h = rec + inp
        y = self.C(self.h)

        # Derivative calculation
        self.dh = (self.h - h) / self.delta
        self.h.retain_grad()

        return y


class BasicPredictor(torch.nn.Module):
    """Simple Predictive Network with Tanh Nonlinearity

    Args:
        y_dim: Input observation dimension
        d_dim: Prediction dimension
        h_dim: Hidden state dimension
    """

    def __init__(self, y_dim: int, d_dim: int, h_dim: int):
        super(BasicPredictor, self).__init__()
        # System matrices
        self.A = torch.nn.Linear(h_dim, h_dim, bias=False)
        self.B = torch.nn.Linear(y_dim, h_dim, bias=False)
        self.C = torch.nn.Linear(h_dim, d_dim, bias=False)

        # Hidden state
        self.h = torch.randn(1, h_dim)
        self.h_init = self.h.clone()

    def forward(self, y: torch.Tensor, first: bool = False) -> torch.Tensor:
        """Simple nonlinear prediction step"""
        if first:
            self.h = self.h_init

        # State update
        h = self.A(self.h) + self.B(y)
        d = self.C(torch.tanh(h))  # Nonlinear projection

        # Detach state for next iteration
        self.h = h.detach()

        return d

class BasicImagePredictor(torch.nn.Module):

    def __init__(self, d_dim: int):
        super(BasicImagePredictor, self).__init__()

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.CenterCrop(32),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 3 * 3, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, d_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, y, first=False):
        return self.net(self.transforms(y))


class BasicTokenGenerator(torch.nn.Module):

    def __init__(self, num_emb: int, emb_dim: int, d_dim: int, y_dim: int, h_dim: int,
                 device: torch.device = torch.device("cpu")):
        super(BasicTokenGenerator, self).__init__()

        u_dim = emb_dim
        du_dim = d_dim
        self.embeddings = torch.nn.Embedding(num_emb, emb_dim)

        self.A = torch.nn.Linear(h_dim, h_dim, bias=False, device=device)
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True)  # initial state (first dimension is batch dim)
        self.h_init = self.h.clone()
        self.dh = torch.zeros_like(self.h)
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = 1.     # already defined in discrete time

    @torch.no_grad()
    def adjust_eigs(self, delta=0.01):
        pass

    def forward(self, u, du, first=False):
        if u is None:
            u = torch.zeros((1, self.u_dim), dtype=torch.float32, device=self.device)
        else:
            u = self.embeddings(u)  # added this
        if du is None:
            du = torch.zeros((1, self.du_dim), dtype=torch.float32, device=self.device)
        h = self.h.detach()
        if first:
            h = self.h_init
        self.h = self.A(h) + self.B(torch.cat([du, u], dim=1))
        y = self.C(torch.tanh(self.h))
        self.dh = (self.h - h) / self.delta
        self.h.retain_grad()
        return y


class BasicTokenPredictor(torch.nn.Module):

    def __init__(self, num_emb: int, emb_dim: int, d_dim: int,  h_dim: int):
        super(BasicTokenPredictor, self).__init__()

        y_dim = emb_dim
        self.embeddings = torch.nn.Embedding(num_emb, emb_dim)

        self.A = torch.nn.Linear(h_dim, h_dim, bias=False)
        self.B = torch.nn.Linear(y_dim, h_dim, bias=False)
        self.C = torch.nn.Linear(h_dim, d_dim, bias=False)
        self.h = torch.randn((1, h_dim))  # initial state
        self.h_init = self.h.clone()

    def forward(self, y, first=False):
        if first:
            self.h = self.h_init
        y = self.embeddings(y)  # added this
        h = self.A(self.h) + self.B(y)
        d = self.C(torch.tanh(h))
        self.h = h.detach()
        return d
