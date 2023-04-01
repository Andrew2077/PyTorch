import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(
        self, channel_num, filters_num
    ) -> None:  # * channel_num: 3, filters_num is the number of filters in the first layer
        super(Discriminator, self).__init__()

        # * Input: (N, channel_num, 64, 64)
        self.disc = nn.Sequential(
            nn.Conv2d(
                channel_num,
                filters_num,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(0.2),
            # * Input: (N, filters_num, 32, 32)
            self._block(filters_num, filters_num * 2, 4, 2, 1),
            # * Input: (N, filters_num*2, 16, 16)
            self._block(filters_num * 2, filters_num * 4, 4, 2, 1),
            # * Input: (N, filters_num*8, 8, 8)
            self._block(filters_num * 4, filters_num * 8, 4, 2, 1),
            # * Input: (N, filters_num*8, 4, 4)
            nn.Conv2d(
                filters_num * 8, 1, kernel_size=4, stride=2, padding=0, bias=False
            ),
            # * Output: (N, 1, 1, 1)
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,  # * No bias in batch norm
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channel_num, filters_num) -> None:
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # * Input: (N, z_dim, 1, 1)
            self._block(z_dim, filters_num * 16, 4, 1, 0),
            # * Input: (N, filters_num*16, 4, 4)
            self._block(filters_num * 16, filters_num * 8, 4, 2, 1),
            # * Input: (N, filters_num*8, 8, 8)
            self._block(filters_num * 8, filters_num * 4, 4, 2, 1),
            # * Input: (N, filters_num*4, 16, 16)
            self._block(filters_num * 4, filters_num * 2, 4, 2, 1),
            # self._block(filters_num*2, filters_num, 4, 2, 1),
            # * Input: (N, filters_num*2, 32, 32)
            nn.ConvTranspose2d(
                in_channels=filters_num * 2,
                out_channels=channel_num,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            # * Output: (N, channel_num, 64, 64)
            nn.Tanh(),  # * [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,  # * No bias in batch norm
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)

def initializing_weights(model):
    for m in model.modules():
        if isinstance(m , (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            #* Initializing weights
            nn.init.normal_(m.weight.data, 0.0, 0.02) #* Normal distribution with mean 0 and std 0.02