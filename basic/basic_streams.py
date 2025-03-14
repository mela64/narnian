import json
import math
import torch
from narnian.streams import Stream
from narnian.attributes import Attributes


class Random(Stream):

    def __init__(self, std: float, shape: tuple[int]):
        super().__init__()
        self.std = std
        self.shape = torch.Size(shape)
        self.attributes = [Attributes((1,), None),
                           Attributes((1,), [self.name.lower()])]
        self.static_d = torch.ones(self.attributes[1].shape)

    def __getitem__(self, step) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        y = self.std * torch.rand((1,) + self.shape)
        d = self.static_d.unsqueeze(0)
        return self.adapt_to_attributes(y, d)


class Sin(Stream):

    def __init__(self, freq: float, phase: float, delta: float):
        super().__init__()
        self.freq = freq
        self.phase = phase
        self.delta = delta
        self.period = 1. / self.freq
        self.attributes = [Attributes((1,), None),
                           Attributes((1,), [self.name.lower()])]
        self.static_d = torch.ones((1, 1))

    def __getitem__(self, step) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        t = step * self.delta + self.phase * self.period
        y = torch.sin(torch.tensor([[2. * math.pi * self.freq * t]]))
        d = self.static_d
        return self.adapt_to_attributes(y, d)


class Square(Stream):

    def __init__(self, freq: float, ampl: float, phase: float, delta: float):
        super().__init__()
        self.freq = freq
        self.ampl = ampl
        self.phase = phase
        self.delta = delta
        self.period = 1. / self.freq
        self.attributes = [Attributes((1,), None),
                           Attributes((1,), [self.name.lower()])]
        self.static_d = torch.ones((1, 1))

    def __getitem__(self, step) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        t = step * self.delta + self.phase * self.period
        y = self.ampl * torch.tensor([[(-1.) ** (math.floor(2. * self.freq * t))]])
        d = self.static_d
        return self.adapt_to_attributes(y, d)


class CombSin(Stream):
    """Synthetic datasets."""

    def __init__(self, f_cap: float | list, c_cap: float | list, order: int, delta: float):
        """Stream obtained by composition of sine waves.

        :param f_cap:       maximum frequency (uniform(0,f_cap)) or list of frequencies.
        :param c_cap:       maximum coefficient (uniform(-c_cap,c_cap)) or list of coefficients.
        :param order:       number of Sine components to sum.
        :param delta:       time (in seconds) between two consecutive elements of the sequence.
        """
        super().__init__()
        if isinstance(f_cap, float):
            self.freqs = f_cap * torch.rand(order)
        elif isinstance(f_cap, list):
            self.freqs = torch.tensor(f_cap)
        else:
            raise Exception(f"expected float or list for f_cap, not {type(f_cap)}")
        self.phases = torch.zeros_like(self.freqs)
        if isinstance(c_cap, float):
            self.coeffs = c_cap * (2 * torch.rand(order) - 1)
        elif isinstance(c_cap, list):
            self.coeffs = torch.tensor(c_cap)
        else:
            raise Exception(f"expected float or list for c_cap, not {type(c_cap)}")

        # check all the dimensions
        assert len(self.coeffs) == len(self.freqs), f"specify the same number of coefficients and frequencies (got {len(self.coeffs)} and {len(self.freqs)} respectively)."
        self.delta = delta
        self.attributes = [Attributes((1,), None),
                           Attributes((1,), [self.name.lower()])]
        self.static_d = torch.ones((1, 1))

        import json
        with open(self.name.lower() + '.json', 'w') as f:
            json.dump({"freqs": str(self.freqs), "coeffs": str(self.coeffs)}, f)

    def __getitem__(self, step) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """returns the input/output pair of sequences and the boolean masks."""
        t = step * self.delta
        y = torch.sum(self.coeffs * torch.sin(2 * math.pi * self.freqs * t + self.phases)).view(1, 1)
        d = self.static_d
        return self.adapt_to_attributes(y, d)
