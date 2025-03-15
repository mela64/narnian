import torch
from typing import Callable
from basic.hl_utils import HL
from narnian.model import Model
from networks.models import _CTBE, _CTE, set_seed
from narnian.attributes import Attributes


class AntisymmetricExpGenerator(_CTE):
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
        super(AntisymmetricExpGenerator, self).__init__(u_shape, d_dim, y_dim, h_dim, delta, sigma, project_every,
                                                        local, device, cnu_memories)

    @torch.no_grad()
    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        # h = torch.randn((1, self.h_init.data.shape[1]))
        # # Antisymmetric matrix construction
        # A = 0.5 * (self.W.weight - self.W.weight.t()) # - 0.1 * torch.eye(self.W.weight.shape[0])
        # A_expm = torch.linalg.matrix_exp(A * self.delta)  # Matrix exponential
        # # Input processing component (constant)
        # A_inv = torch.linalg.inv(A)
        # # inp = A_inv @ (A_expm - self.I) @ self.B(udu).unsqueeze(-1)
        # for _ in range(200):
        #     # h = F.linear(h, A_expm, self.W.bias) + inp.squeeze(-1)
        #     h = F.linear(h, A_expm, self.W.bias)  # + self.B(udu)
        # return h
        return self.B(udu).detach() / torch.sum(udu)  # this is the init

    @staticmethod
    def handle_inputs(du, u):
        return torch.zeros_like(du), torch.zeros_like(u)


class BlockExpGenerator(_CTBE):
    def __init__(self, u_shape, d_dim, y_dim, h_dim, delta, local, cnu_memories: int = 0):
        super().__init__(u_shape=u_shape, d_dim=d_dim, y_dim=y_dim, h_dim=h_dim, delta=delta, local=local,
                         cnu_memories=cnu_memories)

    @torch.no_grad()
    def init_h(self, udu: torch.Tensor) -> torch.Tensor:
        return self.B(udu).detach() / torch.sum(udu)  # this is the init

    @staticmethod
    def handle_inputs(du, u):
        return torch.zeros_like(du), torch.zeros_like(u)


class BasicHLModel(Model):

    def __init__(self, attributes: list[Attributes], delta: float = None,
                 device: torch.device = torch.device("cpu"), seed: int = -1, cnu_memories: int = 0):
        """Creates a model composed of a generator and a predictor."""
        assert delta is not None, f"delta should be specified."
        self.delta = delta
        set_seed(seed)

        # getting shape info from attributes (it is needed to build the generator/predictor)
        assert len(attributes) == 2, "Only two attributes are supported/expected (about y and d)"
        u_shape = attributes[0].shape
        d_dim = attributes[1].shape.numel()
        y_dim = attributes[0].shape.numel()

        #generator = AntisymmetricExpGenerator(u_shape=u_shape, d_dim=d_dim, y_dim=y_dim, h_dim=50,
        #                                      delta=delta, local=True, project_every=0, cnu_memories=cnu_memories)
        generator = BlockExpGenerator(u_shape=u_shape, d_dim=d_dim, y_dim=y_dim, h_dim=1000, delta=delta,
                                      local=True, cnu_memories=cnu_memories)
        predictor = None
        super(BasicHLModel, self).__init__(generator, predictor, attributes, device=device)

        # HL based optimization of the generator
        self.hl_optim = HL(self.generator, gamma=1., theta=0.2, beta=0.01,  # gold: theta=0.2, beta=0.01
                           reset_neuron_costate=False, reset_weight_costate=False, local=True)
        self.loss_gen = torch.nn.functional.mse_loss

    def learn(self,
              y: torch.Tensor | None, yhat: torch.Tensor | None,
              d: torch.Tensor | None, dhat: torch.Tensor | None) \
            -> tuple[float, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Learn from different types of data (some of them could be None)."""

        # clean arguments, ensuring that what should be forced to None is actually forced to None
        _, y, yhat, d, dhat = super().learn(y, yhat, d, dhat)  # it seems unuseful, but IT MUST be called!
        loss_as_float = 0.

        # HL on the generator
        ham = self.hl_optim.compute_hamiltonian(self.loss_gen(y, yhat) if yhat is not None else 0.)
        loss_as_float += ham.item()
        ham.backward()
        self.hl_optim.step()
        self.hl_optim.zero_grad()

        return loss_as_float, y, yhat, d, dhat
