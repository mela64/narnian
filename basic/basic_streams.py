import math
import torch
from narnian.streams import Stream


class Sin(Stream):

    def __init__(self, freq: float, phase: float, delta: float):
        super().__init__()
        self.freq = freq
        self.phase = phase
        self.delta = delta
        self.period = 1. / self.freq

    def __getitem__(self, step) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if step == -1:
            step = self.k
        t = step * self.delta + self.phase * self.period
        y = torch.sin(torch.tensor([[2. * math.pi * self.freq * t]]))
        d = torch.nn.functional.one_hot(torch.LongTensor([self.id]),
                                        num_classes=len(Stream.registered_streams)).to(torch.float32)
        return y, d


class Square(Stream):

    def __init__(self, freq: float, phase: float, delta: float):
        super().__init__()
        self.freq = freq
        self.phase = phase
        self.delta = delta
        self.period = 1. / self.freq

    def __getitem__(self, step) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if step == -1:
            step = self.k
        t = step * self.delta + self.phase * self.period
        y = torch.tensor([[(-1.) ** (math.floor(2. * self.freq * t))]])
        d = torch.nn.functional.one_hot(torch.LongTensor([self.id]),
                                        num_classes=len(Stream.registered_streams)).to(torch.float32)
        return y, d


class CombSin(Stream):
    """Synthetic datasets."""

    def __init__(self, f_cap: float, c_cap: float, order: int, delta: float):
        """Stream obtained by composition of sine waves.

        :param f_cap:       maximum frequency (uniform(0,f_cap)).
        :param c_cap:       maximum coefficient (uniform(-c_cap,c_cap)).
        :param order:       number of Sine components to sum.
        :param delta:       time (in seconds) between two consecutive elements of the sequence.
        """
        super().__init__()
        self.freqs = f_cap * torch.rand(order)
        self.phases = math.pi * (2 * torch.rand(order) - 1)
        self.coeffs = c_cap * (2 * torch.rand(order) - 1)
        self.delta = delta

    def __getitem__(self, step) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """returns the input/output pair of sequences and the boolean masks."""
        step += self.k
        t = step * self.delta
        y = torch.sum(self.coeffs * torch.sin(2 * math.pi * self.freqs * t + self.phases))
        d = torch.nn.functional.one_hot(torch.LongTensor([self.id]),
                                        num_classes=len(Stream.registered_streams)).to(torch.float32)
        return y, d
