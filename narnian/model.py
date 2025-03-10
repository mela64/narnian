import torch
from .streams import Attributes


class Model(torch.nn.Module):
    GEN_AND_PRED = 0  # generate signal and predict descriptor
    GEN = 1  # generate signal (descriptor is given)
    PRED = 2  # predict descriptor (signal is given)
    LEARN_GEN_AND_PRED = 3  # learn to generate signal and to predict descriptor
    LEARN_GEN = 4  # learn to generate signal (nothing is done about the descriptor)
    LEARN_PRED = 5  # learn to predict descriptor (nothing is done about the signal)

    FORWARD_MODES = [GEN_AND_PRED, GEN, PRED]
    LEARN_MODES = [LEARN_GEN_AND_PRED, LEARN_GEN, LEARN_PRED]

    def __init__(self, generator: torch.nn.Module | None, predictor: torch.nn.Module | None,
                 attributes: list[Attributes]):
        """Creates a model composed of a generator and a predictor."""
        super(Model, self).__init__()
        assert generator is not None or predictor is not None, "Both generator and predictor not provided (None, None)"
        self.generator = generator
        self.predictor = predictor
        self.attributes = attributes
        assert len(self.attributes) == 2, f"Expecting two sets of attributes, got {len(attributes)}"

    def forward(self,
                u: torch.Tensor | None, du: torch.Tensor | None,
                y: torch.Tensor | None, d: torch.Tensor | None,
                first: bool = False):
        """Generate a signal and/or predict a descriptor, supporting all the defined modes in Model.FORWARD_MODES.

        Modes:
            y, d = forward(u=GIVEN|None, du=GIVEN|None, y=None, d=None)  -> y=PREDICTED, d=PREDICTED [GEN_AND_PRED]
            y, d = forward(u=GIVEN|None, du=GIVEN|None, y=None, d=GIVEN) -> y=PREDICTED, d=d         [GEN]
            y, d = forward(u=None, du=None, y=GIVEN, d=None)             -> y=y,         d=PREDICTED [PRED]
        """
        mode = Model.detect_forward_mode(du, u, y, d)

        if mode == Model.GEN_AND_PRED:
            y = self.generator.forward(u, du, first) if self.generator is not None else None
            d = self.predictor.forward(y, first) if self.predictor is not None else None
            return y, d
        elif mode == Model.GEN:
            y = self.generator.forward(u, du, first) if self.generator is not None else None
            return y, d
        elif mode == Model.PRED:
            d = self.predictor.forward(y, first) if self.predictor is not None else None
            return y, d

    @staticmethod
    def detect_forward_mode(u: torch.Tensor | None, du: torch.Tensor | None,
                            y: torch.Tensor | None, d: torch.Tensor | None) -> int:
        """Detect what kind of forward the model is asked for."""

        if (u is not None or u is None) and (du is not None or du is None) and y is None and d is None:
            mode = Model.GEN_AND_PRED
        elif (u is not None or u is None) and (du is not None or du is None) and y is None and d is not None:
            mode = Model.GEN
        elif du is None and u is None and y is not None and d is None:
            mode = Model.PRED
        else:
            raise ValueError(f"Unable to detect the forward mode")
        return mode

    def learn(self,
              y: torch.Tensor | None, yhat: torch.Tensor | None,
              d: torch.Tensor | None, dhat: torch.Tensor | None) \
            -> tuple[float, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Learn from different types of data, supporting all the defined modes in Model.LEARN_MODES.

        Modes:
            y, d = learn(y=GIVEN, y_hat=GIVEN, d=GIVEN, d_hat=GIVEN)      -> [LEARN_GEN_AND_PRED]
            y, d = learn(y=GIVEN, y_hat=GIVEN, d=GIVEN|None, d_hat=None)  -> [LEARN_GEN]
            y, d = learn(y=GIVEN|None, y_hat=None, d=GIVEN, d_hat=GIVEN)  -> [LEARN_PRED]
        """
        mode = Model.detect_learn_mode(y, yhat, d, dhat)

        # this "father" method only ensures that whatever could be None is forced to None (i.e., it "cleans" arguments)
        loss_as_float = -1.
        y_clean, yhat_clean, d_clean, dhat_clean = y, yhat, d, dhat

        if mode == Model.LEARN_GEN_AND_PRED:
            pass
        elif mode == Model.LEARN_GEN:
            d_clean = None
        elif mode == Model.LEARN_PRED:
            y_clean = None

        return loss_as_float, y_clean, yhat_clean, d_clean, dhat_clean

    @staticmethod
    def detect_learn_mode(y: torch.Tensor | None, yhat: torch.Tensor | None,
                          d: torch.Tensor | None, dhat: torch.Tensor | None) -> int:
        """Detect what kind of learning the model is asked for."""

        if y is not None and yhat is not None and d is not None and dhat is not None:
            mode = Model.LEARN_GEN_AND_PRED
        elif y is not None and yhat is not None and (d is None or d is not None) and dhat is None:
            mode = Model.LEARN_GEN
        elif (y is None or y is not None) and yhat is None and d is not None and dhat is not None:
            mode = Model.LEARN_PRED
        else:
            raise ValueError(f"Unable to detect the forward mode")
        return mode

    # noinspection PyMethodMayBeStatic
    def compare(self, a: torch.Tensor, b: torch.Tensor, how: str = "mse") -> float:
        """Compare two samples of signals or descriptors, returning a dissimilarity score >= 0."""

        assert how in ['mse', 'max'] or how.startswith("geq"), f"Invalid comparison in terms of {how}"

        if a.dtype == torch.long and b.dtype == torch.long:  # token IDS
            return 1. - float((a == b).item())  # accuracy
        elif how == "mse":
            ret = torch.nn.functional.mse_loss(a, b, reduction='mean')
        elif how == "max":
            ret = 1. - float((torch.argmax(a) == torch.argmax(b)).item())
        else:
            thres = float(how[3:])
            ret = 1. - float(torch.sum((a > thres) == (b > thres)).item()) / a.numel()
        return ret

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        self.load_state_dict(torch.load(file))


class EmptyModel(Model):
    def __init__(self):
        super(EmptyModel, self).__init__(generator=torch.nn.Identity(), predictor=torch.nn.Identity(),
                                         attributes=[Attributes(shape=None, labels=None),
                                                     Attributes(shape=None, labels=None)])
