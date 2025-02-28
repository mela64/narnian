from narnian.model import Model
from networks.models import *
from basic.hl_utils import HL


class BasicHLModel(Model):

    def __init__(self, lr: float = 0.0001, delta: float = None):
        """Creates a model composed of a generator and a predictor."""
        assert delta is not None, f"delta should be specified."
        self.delta = delta
        super(BasicHLModel, self).__init__(AntisymmetricExpGenerator(u_dim=1, du_dim=3, y_dim=1, h_dim=10, delta=delta),
                                           BasicPredictor(y_dim=1, d_dim=3, h_dim=3))

        # SGD based optimization of the predictor
        self.optim = torch.optim.SGD(list(self.predictor.parameters()), lr=lr)
        self.loss_pred = torch.nn.functional.mse_loss

        # HL based optimization of the generator
        self.hl_optim = HL(self.generator, gamma=1., theta=0., beta=1.,
                           reset_neuron_costate=True, reset_weight_costate=True)
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
