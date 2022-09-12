from typing import Union, Tuple
import torch
import torch.nn as nn
import os
# import sys
import numpy as np


class DoubleConv(nn.Module):
    """
    Double convolution class
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: Union[int, str] = "same",
                 activation: str = "ReLU",
                 ):
        act = nn.ReLU()
        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        elif activation.lower == "elu":
            act = nn.ELU()
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=True),
            nn.BatchNorm3d(out_channels),
            act,
            nn.Conv3d(out_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=True),
            nn.BatchNorm3d(out_channels),
            act,
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvTranspose(nn.Module):
    """
    Double convolution class
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: Union[int, str] = "same",
                 activation: str = "ReLU",
                 ):
        act = nn.ReLU()
        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        elif activation.lower == "elu":
            act = nn.ELU()
        if padding == "same" and kernel_size % 2 == 0:
            raise TypeError("You can't provide an even kernel size"
                            " expecting a same padding."
                            )
        padding = int((kernel_size-1)/2)
        super(DoubleConvTranspose, self).__init__()
        self.double_conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=True),
            nn.BatchNorm3d(out_channels),
            act,
            nn.ConvTranspose3d(out_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=True),
            nn.BatchNorm3d(out_channels),
            act,
        )

    def forward(self, x):
        return self.double_conv(x)

# filters,
# kernel_size,


class Unet(nn.Module):
    """
    Implementation of the U-Net used in Lapeyre et al.
    """

    def __init__(self,
                 kernel_size: int = 3,
                 padding: Union[int, str] = "same",
                 maxpool_kernel: Union[int, Tuple[int]] = 2
                 ):
        super().__init__()
        upsampling_kernel = maxpool_kernel
        self.down1 = DoubleConv(in_channels=1,
                                out_channels=32,
                                kernel_size=kernel_size,
                                padding=padding,
                                activation="ReLU"
                                )
        self.down2 = DoubleConv(in_channels=32,
                                out_channels=64,
                                kernel_size=kernel_size,
                                padding=padding,
                                activation="ReLU"
                                )

        self.down3 = DoubleConv(in_channels=64,
                                out_channels=128,
                                kernel_size=kernel_size,
                                padding=padding,
                                activation="ReLU"
                                )

        self.up1 = DoubleConvTranspose(in_channels=128,
                                       out_channels=64,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       activation="ReLU"
                                       )
        self.up2 = DoubleConvTranspose(in_channels=64,
                                       out_channels=32,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       activation="ReLU"
                                       )

        if padding == "same" and kernel_size % 2 == 0:
            raise TypeError("You can't provide an even kernel size"
                            " expecting a same padding."
                            )
        padding = int((kernel_size-1)/2)
        self.last = nn.Sequential(
            nn.ConvTranspose3d(in_channels=32,
                               out_channels=1,
                               kernel_size=kernel_size,
                               padding=padding),
            nn.ReLU(),
        )

        self.pooling = nn.MaxPool3d(maxpool_kernel)
        self.upsampling = nn.Upsample(scale_factor=upsampling_kernel)

    @staticmethod
    def expand_zeros(x, y):
        """
        Expands the x element to have the same shape in the `dim` direction

        y must be a 5 dimentional tensor with the same or more dimentions in
        its 3 last dimentions
        """
        diff = abs(x.shape[2] - y.shape[2])
        if diff:
            x = torch.cat(
                (x,
                 torch.zeros(*x.shape[:-3], diff, x.shape[-2], x.shape[-1])),
                dim=2)
        diff = abs(x.shape[3] - y.shape[3])
        if diff:
            x = torch.cat(
                (x, torch.zeros(*x.shape[:-3], diff, x.shape[-1])), dim=3)
        diff = abs(x.shape[4] - y.shape[4])
        if diff:
            x = torch.cat(
                (x, torch.zeros(*x.shape[:-1], diff)), dim=4)

        return x

    def forward(self, x):
        out_conv1 = self.down1(x)        # After 1st double conv

        x = self.pooling(out_conv1)
        out_conv2 = self.down2(x)        # After 2nd double conv

        x = self.pooling(out_conv2)
        out_conv3 = self.down3(x)        # After 3rd double conv

        upsampled1 = self.upsampling(out_conv3)

        # Concatenation along channel dim
        concat1 = torch.cat((upsampled1, out_conv2), dim=1)
        out_convtranspose1 = self.up1(concat1)
        upsampled2 = self.upsampling(out_convtranspose1)

        concat2 = torch.cat((upsampled2, out_conv1))

        out_convtranspose2 = self.up2(concat2)
        out = self.last(out_convtranspose2)

        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_par_test():
    net = Unet()
    print(count_parameters(net))


def test():
    # Initializing a tensor of shape (8, 8, 8) we can coarsen 3 times
    test_t = torch.randn(1, 1, 2**3, 2**3, 2**3)
    print(f"Input shape: {test_t.shape}")
    net = Unet(kernel_size=3, padding="same", maxpool_kernel=2)
    out = net(test_t)

    print(f"Output shape: {out.shape}")

    assert test_t.shape == out.shape, "Shapes aren't identical"


def test_dns():
    dns_180_path = os.path.join("/home/zatout/Documents/These_yanis/",
                                "Reunion_10012022/"
                                "dns_180/dns_lata_1.sauv.lata.0.RHO")

    sizes = (384, 384, 266)
    rho = np.fromfile(dns_180_path, dtype=np.float32)
    rho = rho.reshape(sizes, order="F")
    print(f"Rho shape: {rho.shape}")
    rho = torch.tensor(rho, dtype=torch.float32)
    rho = rho[None, None, ...]
    net = Unet(kernel_size=3, padding="same", maxpool_kernel=2)
    recon = net(rho)
    print(f"Reconstructed : {recon.shape}")


if __name__ == "__main__":
    count_par_test()
    # test()
    test_dns()
