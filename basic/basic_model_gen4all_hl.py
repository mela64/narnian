import torch
from narnian.model import Model
from basic.utils.hl_utils import HL
from narnian.attributes import Attributes
from modules.networks import set_seed, GenCTBEInitStateBZeroInput


class BasicHLModel(Model):

    def __init__(self, attributes: list[Attributes], delta: float = None,
                 device: torch.device = torch.device("cpu"), seed: int = -1, cnu_memories: int = 0):
        """Creates a model composed of a generator and a predictor."""

        assert delta is not None, f"Parameter 'delta' must be specified (it cannot be None)."
        self.delta = delta
        set_seed(seed)

        # getting shape info from attributes (it is needed to build the generator/predictor)
        assert len(attributes) == 2, "Only two attributes are supported/expected (about y and d)"
        u_shape = attributes[0].shape
        d_dim = attributes[1].shape.numel()
        y_dim = attributes[0].shape.numel()

        # can be one of: { BlockExpGenerator, AntisymmetricExpGenerator }
        generator = GenCTBEInitStateBZeroInput(u_shape=u_shape, d_dim=d_dim, y_dim=y_dim, h_dim=1000, delta=delta,
                                               local=True, cnu_memories=cnu_memories)
        predictor = None
        super(BasicHLModel, self).__init__(generator, predictor, attributes, device=device)

        # HL based optimization of the generator
        self.hl_optim = HL(self.generator, gamma=1., theta=0.2, beta=0.01,
                           reset_neuron_costate=False, reset_weight_costate=False, local=True)
        self.loss_gen = torch.nn.functional.mse_loss

    def learn(self,
              y: torch.Tensor | None, yhat: torch.Tensor | None,
              d: torch.Tensor | None, dhat: torch.Tensor | None) \
            -> tuple[float, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Learn from different types of data (some of them could be None)."""

        # clean arguments, ensuring that what should be forced to None is actually forced to None
        _, y, yhat, d, dhat = super().learn(y, yhat, d, dhat)  # it seems unuseful, but IT MUST be called!

        # step of Hamiltonian Learning
        ham = self.hl_optim.compute_hamiltonian(self.loss_gen(y, yhat))
        ham.backward()
        self.hl_optim.step()
        loss_as_float = ham.item()
        self.hl_optim.zero_grad()

        return loss_as_float, y, yhat, d, dhat
