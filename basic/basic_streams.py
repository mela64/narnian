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
        return (torch.sin(torch.tensor([[2. * math.pi * self.freq * t]])),
                torch.nn.functional.one_hot(torch.LongTensor([self.id]),
                                            num_classes=len(Stream.registered_streams)).to(torch.float32))


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
        return (torch.tensor([[(-1.) ** (math.floor(2. * self.freq * t))]]),
                torch.nn.functional.one_hot(torch.LongTensor([self.id]),
                                            num_classes=len(Stream.registered_streams)).to(torch.float32))
