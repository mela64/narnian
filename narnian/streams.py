import math
import torch
import datetime


class Stream(torch.utils.data.Dataset):
    registered_streams = {}
    __registration_id = 0

    def __init__(self):
        self.name = "unk"
        self.id = -1
        self.enabled = True
        self.k = 0
        self.ctime = datetime.datetime.now()
        self.meta = "none"
        self.creator = "unk"

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def set_name(self, name: str):
        assert name.find('_') == -1, "Underscores are not allowed in name string"
        self.name = name

    def set_meta(self, meta: str):
        assert meta.find('_') == -1, "Underscores are not allowed in meta string"
        self.meta = meta

    def set_step(self, k: int):
        assert k >= 0, f"Invalid step {k} (it must be >= 0)"
        self.k = k

    def next_step(self):
        self.k += 1

    def set_creator(self, creator: str):
        self.creator = creator

    def get_hash(self):
        return Stream.build_hash(self.name, self.creator)

    def get_description_string(self):
        return (f"name=[{self.name}]_creator=[{self.creator}]"
                f"_step=[{self.k}]_ctime=[{str(self.ctime)}]_meta=[{self.meta}]")

    def __getitem__(self, step: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        raise NotImplementedError("Subclasses must implement __getitem__")

    def __str__(self) -> str:
        return "Stream: " + self.name + " [registration id: " + str(self.id) + "]"

    def __len__(self):
        return math.inf

    @staticmethod
    def build_hash(name: str, creator: str | None = None):
        if creator is None:
            creator = "unk"
        assert name.find(':') == -1, "Two dots : cannot be used in stream name"
        assert creator.find(':') == -1, "Two dots : cannot be used in stream creator"
        return f"{creator}:{name}"

    @staticmethod
    def register(name: str, stream_class: object, stream_class_args: dict):
        if name in Stream.registered_streams:
            raise ValueError("Stream name " + name + " already registered")

        Stream.registered_streams[name] = {
            "id": Stream.__registration_id,
            "class": stream_class,
            "class_name": stream_class.__name__,
            "args": stream_class_args,
        }
        Stream.__registration_id += 1

    @staticmethod
    def create(name: str | None, creator: str | None = None):
        try:
            stream = (Stream.registered_streams[name]["class"](**Stream.registered_streams[name]["args"]))
        except KeyError as _:
            raise ValueError("Unknown stream: " + name) from None
        stream.name = name
        stream.id = Stream.registered_streams[name]["id"]
        if creator is not None:
            stream.creator = creator
        return stream


class PassThroughStream(Stream):

    def __init__(self):
        super().__init__()
        self.data_y = torch.empty(1)
        self.data_d = torch.empty(1)

    def set_data(self, y: torch.Tensor, d: torch.Tensor | None):
        self.data_y = y
        self.data_d = d

    def __getitem__(self, step) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        return self.data_y, self.data_d


class BufferedStream(Stream):

    def __init__(self, max_len: int | None = None, static_descriptor: bool = True):
        super().__init__()
        self.max_len = max_len
        self.last_added_step = -1
        self.last_added_pos = -1
        self.static_descriptor = static_descriptor
        self.__d = None

        if self.max_len is None:
            self.data_y = []
            self.data_d = []
        else:
            assert self.max_len >= 1, f"Invalid max_len for buffered stream: {self.max_len}"
            self.data_y = [torch.empty(1)] * self.max_len
            self.data_d = [torch.empty(1)] * self.max_len

    def append_data(self, y: torch.Tensor, d: torch.Tensor | None):
        if self.max_len is None:
            self.data_y.append(y)
            if not self.static_descriptor or len(self.data_d) == 0:
                self.data_d.append(d)
        else:
            self.last_added_step += 1
            self.last_added_pos = (self.last_added_pos + 1) % self.max_len
            self.data_y[self.last_added_pos] = y
            if not self.static_descriptor or self.last_added_step == 0:
                self.data_d[self.last_added_pos] = d

    def __getitem__(self, step) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if self.max_len is None:
            return self.data_y[step], self.data_d[step] if not self.static_descriptor else self.data_d[0]
        else:
            if step > self.last_added_step or (self.last_added_step - step) >= self.max_len:
                raise ValueError(f"Unavailable step for buffered stream: {step} "
                                 f"(last added step: {self.last_added_step}, max length: {self.max_len})")
            else:
                gap = self.last_added_step - step
                step = (self.last_added_pos - gap) % self.max_len
                return self.data_y[step], self.data_d[step] if not self.static_descriptor else self.data_d[0]

    def __len__(self):
        return len(self.data_y) if self.max_len is None else min(self.last_added_step + 1, self.max_len)


class Random(Stream):

    def __init__(self, std: float, shape: tuple):
        super().__init__()
        self.std = std
        self.shape = torch.Size(shape)

    def __getitem__(self, item) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        return (self.std * torch.rand((1,) + self.shape),
                torch.nn.functional.one_hot(torch.LongTensor([self.id]),
                                            num_classes=len(Stream.registered_streams)).to(torch.float32))


class Empty(Stream):

    def __init__(self):
        super().__init__()

    def __getitem__(self, item) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        return None, None
