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
config_model = [(32, 3, 1), (64, 3, 2), ["B", 1],
                (128, 3, 2), ["B", 2],
                (256, 3, 2), ["B", 8],
                (512, 3, 2), ["B", 8],
                (1024, 3, 2), ["B", 4],  # To this point is Darknet-53
                (512, 1, 1), (1024, 3, 1), "S",  # 52 x 52
                (256, 1, 1), "U",
                (256, 1, 1), (512, 3, 1), "S",
                (128, 1, 1), "U",
                (128, 1, 1), (256, 3, 1), "S"]


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_batch_norm=True,
        **kwargs
    ):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=not use_batch_norm,
            **kwargs
        )
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels
        )
        self.leaky_relu = nn.LeakyReLU(
            negative_slope=0.1
        )
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        if self.use_batch_norm:
            x = self.leaky_relu(self.batch_norm(self.conv(x)))
        else:
            x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers.append(
                nn.Sequential(
                    CNNBlock(
                        in_channels=channels,
                        out_channels=channels // 2,
                        kernel_size=1
                    ),
                    CNNBlock(
                        in_channels=channels // 2,
                        out_channels=channels,
                        kernel_size=3,
                        padding=1
                    )
                )
            )
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x


class ScalePrediction(nn.Module):
    '''
        S1: B x 3 x 13 x 13 x [num_classes + 5]
        S2: B x 3 x 26 x 26 x [num_classes + 5]
        S3: B x 3 x 52 x 52 x [num_classes + 5]
    '''

    def __init__(self, channels, num_classes):
        super(ScalePrediction, self).__init__()
        self.predictor = nn.Sequential(
            CNNBlock(
                in_channels=channels,
                out_channels=2 * channels,
                use_batch_norm=True,
                kernel_size=1
            ),
            CNNBlock(
                in_channels=2 * channels,
                out_channels=3 * (5 + num_classes),
                use_batch_norm=False,
                kernel_size=1
            )
        )
        self.num_classes = num_classes

    def forward(self, x):
        x = self.predictor(x)
        # B x [3 * (num_classes + 5)] x H x W
        x = x.reshape(
            x.shape[0], 3, self.num_classes + 5,
            x.shape[2], x.shape[3]
        )
        # B x 3 x [num_classes + 5] x H x W
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        # B x 3 x H x W x [num_classes + 5]

        return x


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super(YOLOv3, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
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
        for module in config_model:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNNBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=kernel_size // 2))
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(channels=in_channels,
                                            use_residual=True,
                                            num_repeats=num_repeats))
            elif isinstance(module, str):
                if module == "S":
                    layers += [ResidualBlock(channels=in_channels,
                                             use_residual=False,
                                             num_repeats=1),
                               CNNBlock(in_channels=in_channels,
                                        out_channels=in_channels // 2,
                                        use_batch_norm=False, kernel_size=1),
                               ScalePrediction(channels=in_channels // 2,
                                               num_classes=self.num_classes)]
                    in_channels = in_channels // 2
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3

        return layers


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    output = model(x)
    assert output[0].shape == (2, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32, num_classes + 5)
    assert output[1].shape == (2, 3, IMAGE_SIZE // 16, IMAGE_SIZE // 16, num_classes + 5)
    assert output[2].shape == (2, 3, IMAGE_SIZE // 8, IMAGE_SIZE // 8, num_classes + 5)
    print("Success!")
