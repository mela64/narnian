import torch
import torchvision
from typing_extensions import Self


class Attributes:

    def __init__(self, shape: tuple[int] | torch.Size | None, labels: list[str] | None,
                 labeling_rule: str = "max", data_type: str = "misc",
                 inv_img_transform: torchvision.transforms = None):

        # type of data
        assert data_type in ["misc", "img", "token_ids"], "Invalid data type"
        self.data_type = data_type  # can be "misc", "img", "token_ids"

        # generic case
        if self.data_type == 'misc':
            assert inv_img_transform is None, "It does not make sense to specify an inv_img_transform for non-img data"
            self.inv_img_transform = None
            self.shape = torch.Size(shape) if shape is not None else None

        # the case of images
        elif self.data_type == 'img':
            self.inv_img_transform = inv_img_transform
            self.shape = torch.Size(shape) if shape is not None else None  # in this case, shape can be not given

        # the case of text represented by token_ids instead of 1-hot representations
        elif self.data_type == 'token_ids':
            assert inv_img_transform is None, "It does not make sense to specify an inv_img_transform for non-img data"
            self.inv_img_transform = None
            self.shape = torch.Size(shape) if shape is not None else None

        # labels are valid when the shape tells that the array is 1d
        self.labels = labels   # only in case of 1d arrays
        self.labeling_rule = labeling_rule  # can be "max" or "geqX", where "X" is a number
        self.superset_labels = labels
        assert labels is None or len(self.shape) == 1, "Attribute labels can only be specified for 1d arrays"
        assert self.labeling_rule == 'max' or self.labeling_rule.startswith('geq'), "Invalid labeling rule"

        # private stuff
        self._num_labels = len(self.labels) if self.labels is not None else 0
        self._indices_in_superset = None  # it only applies to "labels", hence only in case of 1d arrays
        self._num_superset_labels = 0    # it only applies to "labels", hence only in case of 1d arrays
        assert ((self.data_type == "token_ids" and self.shape == torch.Size((1,)) and self._num_labels > 0) or
                (self.data_type == "img" and self._num_labels == 0) or
                (self.data_type == "misc" and (self.labels is None or self.shape.numel() == self._num_labels))), \
            (f"The number of attribute labels is not coherent with its expected number for "
             f"data_type: {self.data_type}, _num_labels: {self._num_labels}, shape: {self.shape}")

    def __getitem__(self, idx):
        if self.labels is None:
            raise ValueError(f"Cannot retrieve any labels, since they are empty")
        if idx < 0 or idx >= self._num_labels:
            raise ValueError(f"Invalid index {idx} for attribute labels of size {self._num_labels}")
        return self.labels[idx]

    def __len__(self):
        return self._num_labels

    def __iter__(self):
        return iter(self.labels) if self.labels is not None else iter([])

    def __str__(self):
        return (f"[Attributes] data_type: {self.data_type}, shape: {self.shape}, "
                f"labeling_rule: {self.labeling_rule}, labels: {self.labels}, "
                f"inv_img_transform: {self.inv_img_transform}, "
                f"_num_superset_labels: {self._num_superset_labels}, "
                f"_indices_in_superset: {self._indices_in_superset})")

    def merge(self, new_labels: Self):
        if self._num_labels == 0:
            self.labels = []
            self.labeling_rule = new_labels.labeling_rule
            self.data_type = new_labels.data_type
            self.inv_img_transform = new_labels.inv_img_transform

        assert self.data_type == 'misc', f"Cannot merge attributes of type: {self.data_type}"
        assert self.shape is not None and len(self.shape) == 1, f"Expected valid and 1d tensor"
        assert new_labels.labels is not None, f"Can only merge non-empty labels"

        for label in new_labels:
            if label not in self.labels:
                self.labels.append(label)
        self._num_labels = len(self.labels)
        self.shape = torch.Size((self._num_labels,))

    def interleave_with(self, super_set_of_attributes: Self):
        assert self.data_type == 'misc', f"Cannot interleave attributes of type: {self.data_type}"
        assert super_set_of_attributes.labels is not None and self.labels is not None, \
            f"Can only interleave non-empty sets of attribute labels"
        assert len(super_set_of_attributes) >= len(self), f"You must provide a super-set of attribute labels"

        # ensuring it is a super-set of the current labels and finding its position
        indices = []
        for label in self.labels:
            assert label in super_set_of_attributes, \
                f"Cannot find attribute label {label} in (expected) super-set {super_set_of_attributes}"
            indices.append(super_set_of_attributes.labels.index(label))

        if len(indices) == len(super_set_of_attributes):
            same_labels_and_order = True
            for j, i in enumerate(indices):
                if j != i:
                    same_labels_and_order = False
                    break
        else:
            same_labels_and_order = False

        if not same_labels_and_order:
            self._num_superset_labels = len(super_set_of_attributes)
            self._indices_in_superset = torch.tensor(indices, dtype=torch.long)
            self.superset_labels = super_set_of_attributes.labels
        else:
            self._num_superset_labels = 0
            self._indices_in_superset = None

    def check_data(self, data: torch.Tensor):
        if self.labels is not None:
            assert data.ndim == 2, "Only 2d tensors are expected for labeled attributes (1st dimension is batch dim)"
            assert self.data_type != "token_ids" or data.shape[1] == 1, \
                f"Invalid shape {data[0].shape} (we discarded the 1st dimension here) for data of type token_ids"

            assert self.data_type == "token_ids" or data.shape[1] == len(self), (
                f"Expected data with {len(self)} components (ignoring the 1st dimension), "
                f"got {data[0].numel()}")
        else:
            if self.shape is not None:
                assert data.shape[1:] == self.shape, (f"Expected data with shape {self.shape}, got {data.shape[1:]} "
                                                      f"(we discarded the 1st dimension here)")

    def interleave_data(self, data: torch.Tensor):
        if self._num_superset_labels > 0:
            data_superset = torch.zeros((data.shape[0], self._num_superset_labels),
                                        device=data.device, dtype=data.dtype)
            data_superset[:, self._indices_in_superset] = data
            return data_superset
        else:
            return data  # do nothing

    def data_to_text(self, data: torch.Tensor):
        if self.labels is None:
            return None

        assert data.shape[0] == 1, f"Code designed for a batch of only 1 element, got {data.shape[0]}"

        # if super-set labels exists, we look into them, otherwise we keep searching in "self.labels"
        label_set = self.superset_labels if self.superset_labels is not None else self.labels

        if self.data_type == "token_ids":
            text = label_set[data[0][0].item()]
        elif self.data_type == "misc":
            if self.labeling_rule == "max":
                j = torch.argmax(data, dim=1)
                text = label_set[j.item()]  # warning: does not work for mini-batches
            elif self.labeling_rule.startswith("geq"):
                thres = float(self.labeling_rule[3:])
                jj = torch.where(data >= thres)[1]  # warning: does not work for mini-batches
                text = ", ".join(label_set[j] for j in jj.tolist())
            else:
                raise ValueError(f"Unknown data-to-text rule: {self.labeling_rule}")
        else:
            raise ValueError(f"Cannot apply data-to-text rule to attributes of type: {self.data_type}")
        return text
