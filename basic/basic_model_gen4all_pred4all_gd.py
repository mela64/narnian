import torch
from narnian.model import Model
from narnian.attributes import Attributes
from modules.networks import PredRNN
from modules.networks import set_seed, GenCTBEInitStateBZeroInput


class BasicModel(Model):

    def __init__(self, attributes: list[Attributes], h_dim: tuple[int] = (500, 10), delta: float = 0.1,
                 cnu_memories: int = 0, lr: float = 0.0001, device: torch.device = torch.device("cpu"), seed: int = -1):
        """Creates a model composed of a generator and a predictor."""

        # getting shape info from attributes (it is needed to build the generator/predictor)
        assert len(attributes) == 2, "Only two attributes are supported/expected (about y and d)"
        u_shape = attributes[0].shape
        d_dim = attributes[1].shape.numel()
        y_dim = attributes[0].shape.numel()
        set_seed(seed)

        generator = GenCTBEInitStateBZeroInput(u_shape=u_shape, d_dim=d_dim, y_dim=y_dim, h_dim=h_dim[0], delta=delta,
                                               local=False, cnu_memories=cnu_memories)
        predictor = PredRNN(y_dim=y_dim, d_dim=d_dim, h_dim=h_dim[1])

        # creating the model (superclass)
        super(BasicModel, self).__init__(generator, predictor, attributes, device=device)

        # extra stuff
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
