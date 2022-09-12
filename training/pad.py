from typing import List
from .utils import _assert_image_tensor
from torch import Tensor
from torch.nn.functional import pad


def get_dimensions3d(img: Tensor) -> List[int]:
    _assert_image_tensor(img)
    channels = 1 if img.ndim == 2 else img.shape[-3]
    height, width, depth = img.shape[-3:]
    return [channels, height, width, depth]


pad = pad


def crop3d(img: Tensor,
           top: int,
           left: int,
           front: int,
           height: int,
           width: int,
           depth: int) -> Tensor:
    """Crop the given image at specified location and output size.
    If the image is torch Tensor, it is expected
    to have [..., H, W, D] shape, where ... means an arbitrary number of
    leading dimensions. If image size is smaller than output size along
    any edge, image is padded with 0 and then cropped.

    Parameters:
    -----------
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the
            top left corner of the image
        top (int): Vertical component of the top left corner of the crop box
        left (int): Horizontal component of the top left corner of the crop box
        front (int): Depth component of the closest point to the camera when
            looking at the crop box
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        depth (int): Depth of the crop box.

    Returns:
    --------
        PIL Image or Tensor: Cropped image.

    """
    _assert_image_tensor(img)

    _, h, w, d = get_dimensions3d(img)
    right = left + width
    bottom = top + height
    back = front + depth

    if left < 0 or top < 0 or front < 0 or right > w or bottom > h or back > d:
        padding_ltrb = [max(-left, 0),
                        max(-top, 0),
                        max(-front, 0),
                        max(right - w, 0),
                        max(bottom - h, 0),
                        max(back - d, 0)]
        return pad(
            img[..., max(top, 0): bottom,
                max(left, 0): right,
                max(front, 0): back],
            padding_ltrb
        )
    return img[..., top:bottom, left:right, back:front]


if __name__ == "__main__":
    import torch
    img = torch.randn(1, 1, 100, 100, 100)
    cropped = crop3d(img, 0, 0, 0, 16, 16, 16)
    print(cropped.shape)
