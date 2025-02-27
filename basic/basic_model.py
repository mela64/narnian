import torch
from narnian.model import Model
from networks.models import BasicGenerator, BasicPredictor


class BasicModel(Model):

    def __init__(self, lr: float = 0.0001):
        """Creates a model composed of a generator and a predictor."""
        super(BasicModel, self).__init__(BasicGenerator(u_dim=1, du_dim=3, y_dim=1, h_dim=5),
                                         BasicPredictor(y_dim=1, d_dim=3, h_dim=3))

        self.optim = torch.optim.SGD(list(self.generator.parameters()) + list(self.predictor.parameters()), lr=lr)
        self.loss_gen = torch.nn.functional.mse_loss
        self.loss_pred = torch.nn.functional.mse_loss

    def learn(self,
              y: torch.Tensor | None, yhat: torch.Tensor | None,
              d: torch.Tensor | None, dhat: torch.Tensor | None) \
            -> tuple[float, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Learn from different types of data (some of them could be None)."""

        # clean arguments, ensuring that what should be forced to None is actually forced to None
        _, y, yhat, d, dhat = super().learn(y, yhat, d, dhat)  # it seems unuseful, but IT MUST be called!

        # evaluating loss function
        loss = ((self.loss_gen(y, yhat) if yhat is not None else 0.) +
                (self.loss_pred(d, dhat) if dhat is not None else 0.))

        # learning
        self.optim.zero_grad()
        loss_as_float = loss.item()
        loss.backward()
        self.optim.step()

        return loss_as_float, y, yhat, d, dhat
