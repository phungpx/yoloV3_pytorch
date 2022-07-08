import torch
import torch.nn as nn

"""
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride)
Every conv is a same convolution.
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class ConvBNLeakyRelu(nn.Module):
    def __init__(
        self,
        in_channels: int, out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 1, conv_repeat: int = 1,
        resconnect: bool = False, attention: Optional[nn.Module] = None,
    ) -> None:
        super(ConvBNLeakyRelu, self).__init__()
        if conv_repeat < 1:
            raise ValueError('Convolution must be repeated more than one time.')

        self.resconnect = resconnect
        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            *[
                nn.Conv2d(
                    in_channels=out_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.1),
            ] * (conv_repeat - 1)
        )

        self.attention = attention if attention is not None else nn.Identity()

        if self.resconnect:
            self.res_transform = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.convs(x)

        if self.resconnect:
            output += self.res_transform(x)

        output = self.attention(output)

        return output


class ConvBNLeaky(nn.Module):
    def __init__(
        self,
        in_channels: int, out_channels: int, kernel_size: int = 1, use_batch_norm: bool = True, **kwargs):
        super(ConvBNLeaky, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1, stride=1,
                bias=not use_batch_norm,
                **kwargs
            ),
            nn.BatchNorm2d(num_features=out_channels) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.leaky = nn.LeakyReLU(negative_slope=0.1)
        self.use_use_batch_norm = use_batch_norm

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    ConvBNLeakyRelu(channels, channels // 2, kernel_size=1),
                    ConvBNLeakyRelu(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            ConvBNLeakyRelu(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            ConvBNLeakyRelu(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    ConvBNLeaky(
                        in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        ConvBNLeaky(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers
