import torch
from narnian.model import Model
from networks.models import AntisymmetricExpGenerator, BasicPredictor
from basic.hl_utils import HL
from narnian.attributes import Attributes


class BasicHLModel(Model):

    def __init__(self, attributes: list[Attributes], lr: float = 0.0001, delta: float = None,
                 device: torch.device = torch.device("cpu")):
        """Creates a model composed of a generator and a predictor."""
        assert delta is not None, f"delta should be specified."
        self.delta = delta

        # getting shape info from attributes (it is needed to build the generator/predictor)
        assert len(attributes) == 2, "Only two attributes are supported/expected (about y and d)"
        u_shape = attributes[0].shape
        d_dim = attributes[1].shape.numel()
        y_dim = attributes[0].shape.numel()

        super(BasicHLModel, self).__init__(AntisymmetricExpGenerator(u_shape=u_shape, d_dim=d_dim, y_dim=y_dim, h_dim=500, delta=delta, local=True, project_every=0),
                                           BasicPredictor(y_dim=1, d_dim=3, h_dim=3),
                                           attributes, device=device)

        # SGD based optimization of the predictor
        self.optim = torch.optim.SGD(list(self.predictor.parameters()), lr=lr)
        self.loss_pred = torch.nn.functional.mse_loss

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
        loss_as_float = 0.

        # HL on the generator
        ham = self.hl_optim.compute_hamiltonian(self.loss_gen(y, yhat) if yhat is not None else 0.)
        loss_as_float += ham.item()
        ham.backward()
        self.hl_optim.step()
        self.hl_optim.zero_grad()

        # SGD on the predictor
        if dhat is not None:
            loss = self.loss_pred(d, dhat)
            self.optim.zero_grad()
            loss_as_float += loss.item()
            loss.backward()
            self.optim.step()

        return loss_as_float, y, yhat, d, dhat
