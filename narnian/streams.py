import os
import csv
import math
import torch
import datetime
import torchvision
from PIL import Image
from .attributes import Attributes


class Stream(torch.utils.data.Dataset):
    k = -1  # this is the time step signal for all the environment, that is automatically updated for all streams

    def __init__(self):
        self.name = "unk"
        self.id = -1
        self.enabled = True
        self.ctime = datetime.datetime.now()
        self.meta = "none"
        self.creator = "unk"
        self.attributes = [Attributes((0,), None),
                           Attributes((0,), None)]

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    @staticmethod
    def reset_step_index():
        Stream.k = -1

    def set_name(self, name: str):
        assert name.find('_') == -1, "Underscores are not allowed in name string"
        self.name = name

    def set_meta(self, meta: str):
        assert meta.find('_') == -1, "Underscores are not allowed in meta string"
        assert meta.find('_') == -1, "Underscores are not allowed in meta string"
        self.meta = meta

    @staticmethod
    def set_step(k: int):
        assert k >= 0, f"Invalid step {k} (it must be >= 0)"
        Stream.k = k

    @staticmethod
    def next_step():
        Stream.k += 1

    def set_creator(self, creator: str):
        self.creator = creator

    def get_hash(self):
        return Stream.build_hash(self.name, self.creator)

    def get_description_string(self):
        return (f"name=[{self.name}]_creator=[{self.creator}]"
                f"_step=[{Stream.k}]_ctime=[{str(self.ctime)}]_meta=[{self.meta}]")

    def __getitem__(self, step: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        raise NotImplementedError("Subclasses must implement __getitem__")

    def get_since(self, since_what_step: int, data_id: int = 0) -> (
            tuple)[list[int] | None, list[torch.Tensor | None] | None, int, Attributes | None]:
        assert data_id >= 0, f"Invalid get_since request with data_id = {data_id}"

        if Stream.k < 0 or not self.enabled:
            return None, None, -1, None

        step = max(since_what_step, 0)
        num_steps = Stream.k - step + 1
        ret_ks = []
        ret_data = []
        for k in range(0, num_steps):
            _k = since_what_step + k
            data = self[_k]

            if data[data_id] is not None:
                ret_ks.append(_k)
                ret_data.append(data[data_id]) if len(data) > 0 else ret_data.append(data)
        return ret_ks, ret_data, self.k, self.attributes[data_id] if len(self.attributes) > 0 else self.attributes

    def adapt_to_attributes(self, y: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.attributes[0].check_data(y)
        self.attributes[1].check_data(d)
        return self.attributes[0].interleave_data(y), self.attributes[1].interleave_data(d)

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
    def create(name: str, creator: str, stream=None):
        stream.name = name
        stream.creator = creator
        if stream.attributes[1].labels[0] == 'unk' and len(stream.attributes[1]) == 1:
            stream.attributes[1].labels[0] = name
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
        if step != -1:
            raise ValueError("Invalid step index for PassThroughStream: only the last sample is available (step: -1)")
        return self.adapt_to_attributes(self.data_y, self.data_d)


class BufferedStream(Stream):

    def __init__(self, use_static_descriptor: bool = True):
        super().__init__()
        self.last_added_step = -1
        self.last_added_pos = -1
        self.use_static_descriptor = use_static_descriptor
        self.first_k = Stream.k  # birthday
        self.last_k = self.first_k - 1  # last added step
        self.__wrapped_stream = None
        self.data_y = []
        self.data_d = []
        self.text_y = []
        self.text_d = []

    def wrap(self, stream: Stream, steps: int):
        self.__wrapped_stream = stream
        self.attributes = stream.attributes
        self.first_k = max(self.first_k, 0)
        self.last_k = self.first_k + steps - 1
        return self

    def get_first_step_offset_given_current_step(self):
        """Additive offset to go from the current step index (Stream.k) to the first step"""
        return self.first_k - Stream.k

    def get_first_step(self):
        return self.first_k

    def set_first_step(self, first_k):
        self.first_k = first_k
        self.last_k = self.first_k - 1

    def append_data(self, y: torch.Tensor, d: torch.Tensor | None, instantaneous: bool = False):
        assert instantaneous or Stream.k == (self.last_k + 1), \
            "BufferedStream can only store data over a single contiguous time interval"

        # switching from probabilities to token indices
        if self.attributes[0].data_type == "token_ids" and y.shape[1] > 1:
            y = torch.argmax(y, dim=1, keepdim=True)
        if self.attributes[1].data_type == "token_ids" and d.shape[1] > 1:
            d = torch.argmax(d, dim=1, keepdim=True)

        # appending data
        self.data_y.append(y)
        if not self.use_static_descriptor or len(self.data_d) == 0:
            self.data_d.append(d)
        if self.attributes[0].data_type == "token_ids":
            self.text_y.append(self.attributes[0].data_to_text(y))
        if self.attributes[1].data_type == "token_ids":
            self.text_d.append(self.attributes[1].data_to_text(d))

        self.last_k += 1

    def to_text(self, length: int | None = None):
        if self.attributes[0].data_type == "token_ids":
            if length is not None:
                le = max(length // 2, 1)
                text_y = " ".join(self.text_y[0:min(le, len(self.text_y))])
                text_y += (" ... " + (" ".join(self.text_y[max(le, len(self.text_y)-le):]))) \
                    if len(self.text_y) > le else ""
            else:
                text_y = " ".join(self.text_y)
        else:
            text_y = None
        if self.attributes[1].data_type == "token_ids":
            if length is not None:
                le = max(length // 2, 1)
                text_d = " ".join(self.text_d[0:min(le, len(self.text_d))])
                text_d += (" ... " + (" ".join(self.text_d[max(le, len(self.text_d)-le):]))) \
                    if len(self.text_d) > le else ""
            else:
                text_d = " ".join(self.text_d)
        else:
            text_d = None
        return text_y, text_d

    def __getitem__(self, step) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:

        # if out of the buffered interval...
        if step > self.last_k or step < self.first_k:
            return None, None

        # otherwise, let's fix indices
        j = step - self.first_k

        # if not wrapped
        if self.__wrapped_stream is None:
            return self.data_y[j], self.data_d[j] if not self.use_static_descriptor else self.data_d[0]
        else:
            return self.__wrapped_stream[j]

    def __len__(self):
        return self.last_k - self.first_k + 1


class Empty(Stream):

    def __init__(self):
        super().__init__()

    def __getitem__(self, step) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        return None, None


class Dataset(Stream):

    def __init__(self, dataset: torch.utils.data.Dataset, data_shape, label_shape):
        super().__init__()
        self.dataset = dataset
        self.attributes = [Attributes(data_shape, None),
                           Attributes(label_shape, None)]

    def __getitem__(self, step) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        y, d = self.dataset[step]
        if y.ndim == 1:  # e.g., [3] or [1]
            y = y.unsqueeze(0)
        elif y.shape[0] > 1:  # e.g., image [28, 28] or [3, 28, 28], while if [1, 28, 28] this is already fine
            y = y.unsqueeze(0)
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(d, device=y.device)
        if d.ndim == 1:  # e.g., [0] or [1]
            d = d.unsqueeze(0)
        return self.adapt_to_attributes(y, d)


class ImageDataset(Stream):

    def __init__(self, image_dir: str, label_file_csv: str,
                 device: torch.device = None, circular: bool = True, single_class: bool = False):
        super().__init__()
        self.image_dir = image_dir
        self.device = device
        self.circular = circular
        self.transform = torchvision.transforms.ToTensor()
        self.inv_transform = torchvision.transforms.ToPILImage()

        # reading the label file (assume CSV format with lines such as: cat.jpg,cat,mammal,animal)
        self.image_paths = []
        self.labels = []

        class_names = {}
        with open(label_file_csv, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                label = parts[1:]
                for lab in label:
                    class_names[lab] = True
        class_name_to_index = {}
        class_names = list(class_names.keys())

        self.attributes = [Attributes(None, None, data_type="img", inv_img_transform=self.inv_transform),
                           Attributes((len(class_names),), class_names,
                                      labeling_rule="geq0.5" if not single_class else "max")]

        for idx, class_name in enumerate(class_names):
            class_name_to_index[class_name] = idx

        with open(label_file_csv, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                image_name = parts[0]
                label = parts[1:]
                target_vector = torch.zeros(len(class_names), dtype=torch.float32)
                for lab in label:
                    idx = class_name_to_index[lab]
                    target_vector[idx] = 1.
                self.image_paths.append(os.path.join(image_dir, image_name))
                self.labels.append(target_vector)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, step):
        if step >= self.__len__():
            if self.circular:
                step = step % self.__len__()
            else:
                raise ValueError(f"Unavailable step for image dataset stream: {step} "
                                 f"(length: {self.__len__()})")

        image = Image.open(self.image_paths[step]).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        label = self.labels[step].unsqueeze(0).to(self.device)  # multi-label vector for the image
        return self.adapt_to_attributes(image, label)


class Tokens(Stream):

    def __init__(self, tokens_file_csv: str, circular: bool = True):
        super().__init__()
        self.circular = circular
        self.token_ids = []
        self.labels = []

        # reading the label file (assume CSV format with lines such as: token,category_label1,category_label2,etc.)
        class_names = {}
        with open(tokens_file_csv, 'r') as f:
            for line in f:
                parts = next(csv.reader([line], quotechar='"', delimiter=','))
                label = parts[1:]
                for lab in label:
                    class_names[lab] = True
        class_name_to_index = {}
        class_names = list(class_names.keys())

        for idx, class_name in enumerate(class_names):
            class_name_to_index[class_name] = idx

        tokens = []
        with open(tokens_file_csv, 'r') as f:
            for line in f:
                parts = next(csv.reader([line], quotechar='"', delimiter=','))
                tokens.append(parts[0])
                label = parts[1:]
                target_vector = torch.zeros(len(class_names), dtype=torch.float32)
                for lab in label:
                    idx = class_name_to_index[lab]
                    target_vector[idx] = 1.
                self.labels.append(target_vector.unsqueeze(0))

        # vocabulary
        idx = 0
        word2id = {}
        sorted_stream_of_tokens = sorted(tokens)
        for token in sorted_stream_of_tokens:
            if token not in word2id:
                word2id[token] = idx
                idx += 1
        id2word = [""] * len(word2id)
        for _word, _id in word2id.items():
            id2word[_id] = _word

        # convert tokenized text to a list of token IDS
        for token in tokens:
            self.token_ids.append(torch.tensor(word2id[token], dtype=torch.long).view(1, 1))

        self.attributes = [Attributes((1,), id2word, data_type="token_ids"),  # trivial max
                           Attributes((len(class_names),), class_names, labeling_rule="geq0.5")]

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, step):
        if step >= self.__len__():
            if self.circular:
                step = step % self.__len__()
            else:
                raise ValueError(f"Unavailable step for token stream: {step} "
                                 f"(length: {self.__len__()})")
        return self.adapt_to_attributes(self.token_ids[step], self.labels[step])
