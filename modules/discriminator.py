# from torch import nn
# import torch.nn.functional as F
from modules.util import kp2gaussian
# import torch
import paddle.fluid as fluid
import paddle


class DownBlock2d(fluid.dygraph.Layer):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock2d, self).__init__()
        self.conv = fluid.dygraph.Conv2D(num_channels=in_features, num_filters=out_features, filter_size=kernel_size)
        # self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)

        self.sn=sn
        # if sn:
        #     self.conv = fluid.dygraph.SpectralNorm(self.conv)
            # self.conv = nn.utils.spectral_norm(self.conv)

        self.norm=norm
        # if norm:
        #     self.norm = fluid.layers.Instance_Norm(out_features)
        #     # self.norm = nn.InstanceNorm2d(out_features, affine=True)
        # else:
        #     self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.sn:
            out = fluid.layers.spectral_norm(out)
        if self.norm:
            out = fluid.layers.instance_norm(out)
            # out = self.norm(out)
        out = fluid.layers.leaky_relu(out, 0.2)
        # out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = fluid.layers.pool2d(out, pool_size=[2, 2], pool_type='avg')
            # out = F.avg_pool2d(out, (2, 2))
        return out


class Discriminator(fluid.dygraph.Layer):
# class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=False, use_kp=False, num_kp=10, kp_variance=0.01, **kwargs):
        super(Discriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(num_channels + num_kp * use_kp if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn))

        self.down_blocks = down_blocks
        self.conv = fluid.dygraph.Conv2D(num_channels=512, num_filters=1, filter_size=1)
        self.sn=sn
        # if sn:
        #     self.conv = fluid.dygraph.SpectralNorm(self.conv)
        # self.down_blocks = nn.ModuleList(down_blocks)
        # self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        # if sn:
        #     self.conv = nn.utils.spectral_norm(self.conv)
        self.use_kp = use_kp
        self.kp_variance = kp_variance

    def forward(self, x, kp=None):
        feature_maps = []
        out = x
        if self.use_kp:
            heatmap = kp2gaussian(kp, x.shape[2:], self.kp_variance)
            out = fluid.layers.concat([out, heatmap], axis=1)
            # out = torch.cat([out, heatmap], dim=1)

        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)
        if self.sn:
            prediction_map = fluid.layers.spectral_norm(prediction_map)

        return feature_maps, prediction_map

class MultiScaleDiscriminator(fluid.dygraph.Layer):
# class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale (scale) discriminator
    """

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales
        discs = {}
        for scale in scales:
            discs[str(scale).replace('.', '-')] = Discriminator(**kwargs)
        self.discs = discs
        # self.discs = nn.ModuleDict(discs)

    def forward(self, x, kp=None):
        out_dict = {}
        for scale, disc in self.discs.items():
            scale = str(scale).replace('-', '.')
            key = 'prediction_' + scale
            feature_maps, prediction_map = disc(x[key], kp)
            out_dict['feature_maps_' + scale] = feature_maps
            out_dict['prediction_map_' + scale] = prediction_map
        return out_dict
