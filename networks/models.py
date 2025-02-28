import torch
import torch.nn.functional as F


class BasicGenerator(torch.nn.Module):

    def __init__(self, u_dim: int, du_dim: int, y_dim: int, h_dim: int, device: torch.device = torch.device("cpu")):
        super(BasicGenerator, self).__init__()
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


class DiagReal(torch.nn.Module):

    def __init__(self, u_dim: int, du_dim: int, y_dim: int, h_dim: int, device: torch.device = torch.device("cpu")):
        super(DiagReal, self).__init__()
        self.diag = torch.nn.Linear(in_features=1, out_features=h_dim, bias=False, device=device, dtype=torch.float32)
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
        self.diag.weight.copy_(torch.sign(self.diag.weight))

    def forward(self, u, du, first=False):
        if u is None:
            u = torch.zeros((1, self.u_dim), dtype=torch.float32, device=self.device)
        if du is None:
            du = torch.zeros((1, self.du_dim), dtype=torch.float32, device=self.device)
        h = self.h.detach()
        if first:
            h = self.h_init
        self.h = self.diag.weight.view(self.diag.out_features) * h + self.B(torch.cat([du, u], dim=1))
        y = self.C(torch.tanh(self.h))
        self.dh = (self.h - h) / self.delta
        self.h.retain_grad()
        return y


class DiagCompl(torch.nn.Module):

    def __init__(self, u_dim: int, du_dim: int, y_dim: int, h_dim: int, device: torch.device = torch.device("cpu")):
        super(DiagCompl, self).__init__()
        self.diag = torch.nn.Linear(in_features=1, out_features=h_dim, bias=False, device=device, dtype=torch.cfloat)
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device, dtype=torch.cfloat)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device, dtype=torch.cfloat)
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True, dtype=torch.cfloat)  # initial state (first dimension is batch dim)
        self.h_init = self.h.clone()
        self.dh = torch.zeros_like(self.h)
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = 1.     # already defined in discrete time

    @torch.no_grad()
    def adjust_eigs(self, delta=0.01):
        self.diag.weight.div_(self.diag.weight.abs())

    def forward(self, u, du, first=False):
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
        self.h = self.diag.weight.view(self.diag.out_features) * h + self.B(torch.cat([du, u], dim=1))
        y = self.C(torch.tanh(self.h)).real
        self.dh = (self.h - h) / self.delta
        self.h.retain_grad()
        return y


class AntisymmetricExpGenerator(torch.nn.Module):

    def __init__(self, u_dim: int, du_dim: int, y_dim: int, h_dim: int, delta: float,
                 device: torch.device = torch.device("cpu")):
        super(AntisymmetricExpGenerator, self).__init__()
        self.W = torch.nn.Linear(h_dim, h_dim, bias=False, device=device)
        self.I = torch.eye(h_dim, requires_grad=False, device=device)
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True)  # initial state (first dimension is batch dim)
        self.h_init = self.h.clone()
        self.dh = torch.zeros_like(self.h)
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = delta

    @torch.no_grad()
    def adjust_eigs(self, delta=0.01):
        pass

    def forward(self, u, du, first=False):
        if u is None:
            u = torch.zeros((1, self.u_dim), dtype=torch.float32, device=self.device)
        if du is None:
            du = torch.zeros((1, self.du_dim), dtype=torch.float32, device=self.device)
        h = self.h.detach()
        if first:
            h = self.h_init
        A = 0.5 * (self.W.weight - self.W.weight.t())
        A_inv = torch.linalg.inv(A)
        A_expm = torch.linalg.matrix_exp(A * self.delta)
        rec = F.linear(h, A_expm, self.W.bias)
        inp = A_inv @ (A_expm - self.I) @ self.B(torch.cat([du, u], dim=1)).unsqueeze(-1)   # for batch operation
        self.h = rec + inp.squeeze(-1)
        # y = self.C(torch.tanh(self.h))
        y = self.C(self.h)
        self.dh = (self.h - h) / self.delta
        self.h.retain_grad()
        return y


class BlockAntisymmetricGenerator(torch.nn.Module):

    def __init__(self, u_dim: int, du_dim: int, y_dim: int, h_dim: int, delta: float = None,
                 alpha: float = 0., device: torch.device = torch.device("cpu")):
        super(BlockAntisymmetricGenerator, self).__init__()
        assert h_dim % 2 == 0, f"CTB is made of 2x2 blocks, please specify an even hidden dim (provided h_dim={h_dim})."
        self.order = h_dim // 2
        # we define the omegas as a learnable parameter and alphas as a buffer
        self.omega = torch.nn.Parameter(torch.empty(self.order))
        self.ones = torch.ones(self.order, dtype=torch.float32, requires_grad=False)
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)
        if alpha > 0.:
            # in this case we want to add the feedback parameter alpha and use it to move eigenvalues on the unit circle
            self.project_method = 'const'
            self.register_buffer('alpha', torch.full_like(self.omega, alpha, requires_grad=False))
        elif alpha == 0.:
            # this is the case in which we want to divide by the modulus
            self.project_method = 'modulus'
            self.register_buffer('alpha', torch.zeros_like(self.omega, requires_grad=False))
        elif alpha == -1.:
            self.project_method = 'alpha'
            self.register_buffer('alpha', torch.zeros_like(self.omega, requires_grad=False))
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True)  # initial state (first dimension is batch dim)
        self.h_init = self.h.clone()
        self.dh = torch.zeros_like(self.h)
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = delta
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # we simply initialize in uniform(0,1)
        torch.nn.init.uniform_(self.omega)

    @torch.no_grad()
    def adjust_eigs(self, delta):
        with torch.no_grad():
            if self.project_method == 'alpha':
                self.alpha.copy_((1. - torch.sqrt(1. - torch.square(delta * self.omega))) / delta)
            elif self.project_method == 'modulus':
                # todo capire se questa operazione è corretta nel caso tempo variante, mi spiego meglio, il module
                #  è calcolato assumendo alpha=0, poi alpha cambia, forse però non è un problema
                module = torch.sqrt(torch.square(self.ones) + torch.square(self.delta * self.omega))
                self.omega.div_(module)
                self.ones.div_(module)

    def forward(self, u, du, first=False):
        if u is None:
            u = torch.zeros((1, self.u_dim), dtype=torch.float32, device=self.device)
        if du is None:
            du = torch.zeros((1, self.du_dim), dtype=torch.float32, device=self.device)
        h = self.h.detach()
        if first:
            h = self.h_init
        h_pair = h.view(-1, self.order, 2)  # we start by reshaping the state in pairs
        h1 = (self.ones - self.delta * self.alpha) * h_pair[..., 0] + self.delta * self.omega * h_pair[..., 1]
        h2 = -self.delta * self.omega * h_pair[..., 0] + (self.ones - self.delta * self.alpha) * h_pair[..., 1]
        rec = torch.stack([h1, h2], dim=-1).flatten(start_dim=1)
        inp = self.delta * self.B(torch.cat([du, u], dim=1))
        self.h = rec + inp
        # y = self.C(torch.tanh(selfh))
        y = self.C(self.h)
        self.dh = (self.h - h) / self.delta
        self.h.retain_grad()
        return y


class BlockAntisymmetricExpGenerator(torch.nn.Module):

    def __init__(self, u_dim: int, du_dim: int, y_dim: int, h_dim: int, delta: float,
                 device: torch.device = torch.device("cpu")):
        super(BlockAntisymmetricExpGenerator, self).__init__()
        self.xi = None
        assert h_dim % 2 == 0, f"CTBE is made of 2x2 blocks, please specify an even hidden dim (provided h_dim={h_dim})."
        self.order = h_dim // 2
        # we define the omegas as a learnable parameter and alphas as a buffer
        self.omega = torch.nn.Parameter(torch.empty(self.order))
        self.B = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        self.C = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)
        self.h = torch.randn((1, h_dim), device=device, requires_grad=True)  # initial state (first dimension is batch dim)
        self.h_init = self.h.clone()
        self.dh = torch.zeros_like(self.h)
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device
        self.delta = delta
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # we simply initialize in uniform(0,1)
        torch.nn.init.uniform_(self.omega)

    @torch.no_grad()
    def adjust_eigs(self, delta=0.01):
        pass

    def forward(self, u, du, first=False):
        if u is None:
            u = torch.zeros((1, self.u_dim), dtype=torch.float32, device=self.device)
        if du is None:
            du = torch.zeros((1, self.du_dim), dtype=torch.float32, device=self.device)
        h = self.h.detach()
        if first:
            h = self.h_init
        # Compute cos and sin terms once
        cos_t = torch.cos(self.omega * self.delta)
        sin_t = torch.sin(self.omega * self.delta)
        h_pair = h.view(-1, self.order, 2)  # we start by reshaping the state in pairs
        h1 = cos_t * h_pair[..., 0] + sin_t * h_pair[..., 1]
        h2 = -sin_t * h_pair[..., 0] + cos_t * h_pair[..., 1]
        rec = torch.stack([h1, h2], dim=-1).flatten(start_dim=1)
        u_hat = self.B(torch.cat([u, du], dim=1)).view(-1, self.order, 2)  # reshape the projected input in pairs
        inp1 = 1. / self.omega * (sin_t * u_hat[..., 0] - (cos_t - 1) * u_hat[..., 1])
        inp2 = 1. / self.omega * ((cos_t - 1) * u_hat[..., 0] + sin_t * u_hat[..., 1])
        inp = torch.stack([inp1, inp2], dim=-1).flatten(start_dim=1)
        self.h = rec + inp
        # y = self.C(torch.tanh(self.h))
        y = self.C(self.h)
        self.dh = (self.h - h) / self.delta
        self.h.retain_grad()
        return y


class BasicPredictor(torch.nn.Module):

    def __init__(self, y_dim: int, d_dim: int,  h_dim: int):
        super(BasicPredictor, self).__init__()
        self.A = torch.nn.Linear(h_dim, h_dim, bias=False)
        self.B = torch.nn.Linear(y_dim, h_dim, bias=False)
        self.C = torch.nn.Linear(h_dim, d_dim, bias=False)
        self.h = torch.randn((h_dim, h_dim))  # initial state
        self.h_init = self.h.clone()

    def forward(self, y, first=False):
        if first:
            self.h = self.h_init
        h = self.A(self.h) + self.B(y)
        d = self.C(torch.tanh(h))
        self.h = h.detach()
        return d
