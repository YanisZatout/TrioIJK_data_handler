from typing import Union, Tuple
from collections.abc import Sequence
import numbers
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms.functional import get_dimensions
from torchvision.utils import _log_api_usage_once


def select_patch(data: Tensor, patch_size: Union[int, Tuple]):
    """
    From the given data, randomly selects a patch of data in 3D
    Parameters:
    ----------
    data: Tensor
        Data to select the patch from
    patch_size: Union[int, Tuple]
        Size of the patch in each of the 3 directions of space
    Returns:
    ----------
    patch: Tensor
        Patch in 3D space of the shape (..., *patch_size)
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    shape = data.shape
    slices_begin = torch.tensor([0, 0, 0])
    for i, s in enumerate(shape[-3:]):
        slices_begin[i] = torch.randint(
            low=0, high=s - patch_size[i], size=(1,))
    return data[...,
                slices_begin[0]:slices_begin[0] + patch_size[0],
                slices_begin[1]:slices_begin[1] + patch_size[1],
                slices_begin[2]:slices_begin[2] + patch_size[2]
                ]


def _setup_size3(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0], size[0]

    if len(size) != 3:
        raise ValueError(error_msg)

    return size


class RandomCrop3d(torch.nn.Module):

    @staticmethod
    def get_params(
        img: Tensor,
            output_size: Tuple[int, int, int]
    ) -> Tuple[number.Number, number.Number, number.Number, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop``
            for random crop.
        """
        _, _, x, y, z = img.shape
        tx, ty, tz = output_size

        if x + 1 < tx or y + 1 < ty or z + 1 < tz:
            raise ValueError(
                f"Required crop size {(tx, ty, tz)} is larger than"
                " input image size {(x, y, z)}")

        if x == tx and y == ty and z == tz:
            return 0, 0, 0, x, y, z

        i = torch.randint(0, x - tx + 1, size=(1,)).item()
        j = torch.randint(0, y - ty + 1, size=(1,)).item()
        k = torch.randint(0, z - tz + 1, size=(1,)).item()
        return i, j, k, tx, ty, tz

    def __init__(
        self,
            size,
            padding=None,
            pad_if_needed=False,
            fill=0,
            padding_mode="constant"
    ):
        super().__init__()
        _log_api_usage_once(self)

        self.size = tuple(_setup_size3(
            size, error_msg="Please provide only three dimensions (x, y, z)"
            "for size."))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width, space = get_dimensions(img)
        # pad the space if needed
        if self.pad_if_needed and space < self.size[2]:
            padding = [0, 0, self.size[2] - space]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        i, j, k, x, y, z = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"


def test_patch():
    size = 100
    tensor = torch.randn(1, 1, size, size, size)
    patch = select_patch(tensor, 16)
    assert patch.shape[-3:] == (16, 16, 16), "Error while trying to take patch"
    tensor = torch.randn(1, 3, size, size, size)
    patch = select_patch(tensor, 16)
    print(f"{patch.shape}")


if __name__ == "__main__":
    test_patch()
