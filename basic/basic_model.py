import torch
from narnian.model import Model


class BasicGenerator(torch.nn.Module):

    def __init__(self, u_dim: int, du_dim: int, y_dim: int, h_dim: int, device: torch.device = torch.device("cpu")):
        super(BasicGenerator, self).__init__()
        self.linear_hidden_to_hidden = torch.nn.Linear(h_dim, h_dim, bias=False, device=device)
        self.linear_input_to_hidden = torch.nn.Linear(u_dim + du_dim, h_dim, bias=False, device=device)
        self.hidden_to_output = torch.nn.Linear(h_dim, y_dim, bias=False, device=device)
        self.h = torch.randn((1, h_dim), device=device)  # initial state (first dimension is batch dim)
        self.h_init = self.h.clone()
        self.u_dim = u_dim
        self.du_dim = du_dim
        self.device = device

    def forward(self, u, du, first=False):
        if u is None:
            u = torch.zeros((1, self.u_dim), dtype=torch.float32, device=self.device)
        if du is None:
            du = torch.zeros((1, self.du_dim), dtype=torch.float32, device=self.device)
        if first:
            self.h = self.h_init
        h = self.linear_hidden_to_hidden(self.h) + self.linear_input_to_hidden(torch.cat([du, u], dim=1))
        y = self.hidden_to_output(torch.tanh(h))
        self.h = h.detach()
        return y


class BasicPredictor(torch.nn.Module):

    def __init__(self, y_dim: int, d_dim: int,  h_dim: int):
        super(BasicPredictor, self).__init__()
        self.linear_hidden_to_hidden = torch.nn.Linear(h_dim, h_dim, bias=False)
        self.linear_input_to_hidden = torch.nn.Linear(y_dim, h_dim, bias=False)
        self.hidden_to_output = torch.nn.Linear(h_dim, d_dim, bias=False)
        self.h = torch.randn((h_dim, h_dim))  # initial state
        self.h_init = self.h.clone()

    def forward(self, y, first=False):
        if first:
            self.h = self.h_init
        h = self.linear_hidden_to_hidden(self.h) + self.linear_input_to_hidden(y)
        d = self.hidden_to_output(torch.tanh(h))
        self.h = h.detach()
        return d


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
