import torch
from narnian.model import Model
from narnian.attributes import Attributes
from modules.networks import GenRNNTokenLM


class BasicTokenModel(Model):

    def __init__(self, attributes: list[Attributes], emb_dim: int = 16, h_dim: int = 100, lr: float = 0.0001,
                 device: torch.device = torch.device("cpu"), seed: int = -1):
        """Creates a model composed of a generator and a predictor."""

        # getting shape info from attributes (it is needed to build the generator/predictor)
        assert len(attributes) == 2, "Only two attributes are supported/expected (about y and d)"
        num_emb = len(attributes[0])  # the length of the vocabulary
        d_dim = attributes[1].shape.numel()
        y_dim = len(attributes[0])  # the length of the vocabulary

        # creating the model (superclass)
        super(BasicTokenModel, self).__init__(
            generator=GenRNNTokenLM(num_emb=num_emb, emb_dim=emb_dim, d_dim=d_dim, y_dim=y_dim,
                                    h_dim=h_dim, device=device, seed=seed),
            predictor=None,
            attributes=attributes, device=device, seed=seed)

        # extra stuff
        def loss_gen(y, yhat):

            # yhat is expected to be a token index, with shape [1, 1], and we need to convert it to shape [1] for CE
            assert yhat.shape[1] == 1, "Only one label per sample is supported"
            return torch.nn.functional.cross_entropy(y, yhat.view(-1))

        self.optim = torch.optim.SGD(self.generator.parameters(), lr=lr)
        self.loss_gen = loss_gen

    def learn(self,
              y: torch.Tensor | None, yhat: torch.Tensor | None,
              d: torch.Tensor | None, dhat: torch.Tensor | None) \
            -> tuple[float, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Learn from different types of data (some of them could be None)."""

        # clean arguments, ensuring that what should be forced to None is actually forced to None
        _, y, yhat, d, dhat = super().learn(y, yhat, d, dhat)  # it seems unuseful, but IT MUST be called!

        # evaluating loss function
        loss = self.loss_gen(y, yhat)

        # learning
        self.optim.zero_grad()
        loss_as_float = loss.item()
        loss.backward()
        self.optim.step()

        # teaching
        self.generator.y = yhat  # only exploited by LM-like generators

        return loss_as_float, y, yhat, d, dhat
