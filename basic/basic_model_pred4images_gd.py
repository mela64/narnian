import torch
from narnian.model import Model
from modules.networks import set_seed
from narnian.attributes import Attributes
from modules.networks import PredCNN, PredCNNCNU


class BasicImageModel(Model):

    def __init__(self, attributes: list[Attributes], lr: float = 0.0001,
                 device: torch.device = torch.device("cpu"), seed: int = -1):
        """Creates a model composed of a generator and a predictor."""

        # getting shape info from attributes (it is needed to build the generator/predictor)
        assert len(attributes) == 2, "Only two attributes are supported/expected (about y and d)"
        d_dim = attributes[1].shape.numel()
        set_seed(seed)

        # calling constructor
        super(BasicImageModel, self).__init__(None,
                                              PredCNN(d_dim=d_dim, device=device),
                                              attributes, device=device)

        # extra stuff
        self.optim = torch.optim.SGD(self.predictor.parameters(), lr=lr)
        self.loss_pred = torch.nn.functional.binary_cross_entropy

    def learn(self,
              y: torch.Tensor | None, yhat: torch.Tensor | None,
              d: torch.Tensor | None, dhat: torch.Tensor | None) \
            -> tuple[float, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Learn from different types of data (some of them could be None)."""

        # clean arguments, ensuring that what should be forced to None is actually forced to None
        _, y, yhat, d, dhat = super().learn(y, yhat, d, dhat)  # it seems unuseful, but IT MUST be called!

        # evaluating loss function
        loss = self.loss_pred(d, dhat)

        # learning
        self.optim.zero_grad()
        loss_as_float = loss.item()
        loss.backward()
        self.optim.step()

        return loss_as_float, y, yhat, d, dhat


class BasicImageModelCNU(Model):

    def __init__(self, attributes: list[Attributes], cnu_memories: int = 5, lr: float = 0.0001, delta: int = 1,
                 scramble: bool = False,
                 lr_head: float | None = None,
                 device: torch.device = torch.device("cpu"), seed: int = -1) -> None:
        """Creates a model composed of a generator and a predictor."""

        # getting shape info from attributes (it is needed to build the generator/predictor)
        assert len(attributes) == 2, "Only two attributes are supported/expected (about y and d)"
        d_dim = attributes[1].shape.numel()
        set_seed(seed)

        # keeping the same learning rate
        if lr_head is None:
            lr_head = lr

        # calling constructor
        super(BasicImageModelCNU, self).__init__(None,
                                                 PredCNNCNU(d_dim=d_dim, cnu_memories=cnu_memories,
                                                            device=device, delta=delta,
                                                            scramble=scramble),
                                                 attributes, device=device, seed=seed)

        # extra stuff
        self.optim = torch.optim.SGD([
            {'params': self.predictor.net[:-2].parameters(), 'lr': lr},
            {'params': self.predictor.net[-2:].parameters(), 'lr': lr_head}
        ], lr=lr)
        self.loss_pred = torch.nn.functional.binary_cross_entropy

    def learn(self,
              y: torch.Tensor | None, yhat: torch.Tensor | None,
              d: torch.Tensor | None, dhat: torch.Tensor | None) \
            -> tuple[float, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Learn from different types of data (some of them could be None)."""

        # clean arguments, ensuring that what should be forced to None is actually forced to None
        _, y, yhat, d, dhat = super().learn(y, yhat, d, dhat)  # it seems unuseful, but IT MUST be called!

        # evaluating loss function
        loss = self.loss_pred(d, dhat)

        # learning
        self.optim.zero_grad()
        loss_as_float = loss.item()
        loss.backward()
        self.optim.step()

        return loss_as_float, y, yhat, d, dhat
