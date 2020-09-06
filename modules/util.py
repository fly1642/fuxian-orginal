# from torch import nn

# import torch.nn.functional as F
# import torch

# from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
import paddle
import paddle.fluid as fluid
import numpy as np

def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']

    coordinate_grid = make_coordinate_grid(spatial_size)
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = [1,] * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = fluid.layers.reshape(coordinate_grid, shape)
    repeats = mean.shape[:number_of_leading_dimensions] + [1, 1, 1]
    coordinate_grid=coordinate_grid.numpy()
    for i in range(len(repeats)):
        coordinate_grid =np.repeat(coordinate_grid, repeats[i],axis=i)
    coordinate_grid=fluid.dygraph.to_variable(coordinate_grid)
    # coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    # number_of_leading_dimensions = len(mean.shape) - 1
    # shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    # coordinate_grid = coordinate_grid.view(*shape)
    # repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    # coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + [1, 1, 2]
    mean = fluid.layers.reshape(mean, shape)
    # shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    # mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)
    
    out = fluid.dygraph.to_variable(np.sum(-0.5 * (mean_sub.numpy() ** 2),axis=-1))
    out = paddle.fluid.layers.exp(out / kp_variance)
    # out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def make_coordinate_grid(spatial_size):
# def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = fluid.layers.arange(0,w)
    y = fluid.layers.arange(0,h)
    # x = fluid.layers.range(w).type(type)
    # y = fluid.layers.range(h).type(type)
    # x = torch.arange(w).type(type)
    # y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    y=fluid.layers.reshape(y,[-1,1])
    yy=np.repeat(y.numpy(), w, axis=1)
    yy=fluid.dygraph.to_variable(yy)
    x=fluid.layers.reshape(x,[1,-1])
    xx=np.repeat(x.numpy(), h, axis=0)
    xx=fluid.dygraph.to_variable(xx)
    # yy = y.view(-1, 1).repeat(1, w)
    # xx = x.view(1, -1).repeat(h, 1)

    xx=fluid.layers.unsqueeze(xx, axes=2)
    yy=fluid.layers.unsqueeze(yy, axes=2)
    meshed =fluid.layers.concat([xx, yy], 2)
    # meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class ResBlock2d(fluid.dygraph.Layer):
# class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = fluid.dygraph.Conv2D(num_channels=in_features, num_filters=in_features, filter_size=kernel_size,
                               padding=padding)
        self.conv2 = fluid.dygraph.Conv2D(num_channels=in_features, num_filters=in_features, filter_size=kernel_size,
                               padding=padding)
        self.norm1 = fluid.dygraph.BatchNorm(num_channels=in_features)
        self.norm2 = fluid.dygraph.BatchNorm(num_channels=in_features)
        # self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
        #                        padding=padding)
        # self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
        #                        padding=padding)
        # self.norm1 = BatchNorm2d(in_features, affine=True)
        # self.norm2 = BatchNorm2d(in_features, affine=True)
    def forward(self, x):
        out = self.norm1(x)
        out = fluid.layers.relu(out)
        # out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = fluid.layers.relu(out)
        # out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(fluid.dygraph.Layer):
# class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()
        # super(UpBlock2d, self).__init__()

        self.conv = fluid.dygraph.Conv2D(num_channels=in_features, num_filters=out_features, filter_size=kernel_size,
                               padding=padding, groups=groups)
        self.norm = fluid.dygraph.BatchNorm(num_channels=out_features)
        # self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
        #                       padding=padding, groups=groups)
        # self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = fluid.layers.interpolate(x, scale=2)
        # out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = fluid.layers.relu(out)
        # out = F.relu(out)
        return out


class DownBlock2d(fluid.dygraph.Layer):
# class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = fluid.dygraph.Conv2D(num_channels=in_features, num_filters=out_features, filter_size=kernel_size, padding=padding, groups=groups)
        self.norm = fluid.dygraph.BatchNorm(num_channels=out_features)
        self.pool = fluid.dygraph.Pool2D(pool_size=[2,2], pool_stride=2, pool_type='avg')
        # self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
        #                       padding=padding, groups=groups)
        # self.norm = BatchNorm2d(out_features, affine=True)
        # self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = fluid.layers.relu(out)
        # out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(fluid.dygraph.Layer):
# class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = fluid.dygraph.Conv2D(num_channels=in_features, num_filters=out_features, filter_size=kernel_size,
                               padding=padding, groups=groups)
        self.norm = fluid.dygraph.BatchNorm(num_channels=out_features)
        # self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
        #                       kernel_size=kernel_size, padding=padding, groups=groups)
        # self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = fluid.layers.relu(out)
        # out = F.relu(out)
        return out


class Encoder(fluid.dygraph.Layer):
# class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = down_blocks
        # self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(fluid.dygraph.Layer):
# class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = up_blocks
        # self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = fluid.layers.concat([out, skip], 1)
            # out = torch.cat([out, skip], dim=1)
        return out


class Hourglass(fluid.dygraph.Layer):
# class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(fluid.dygraph.Layer):
# class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        
        meshgrids=np.meshgrid(np.arange(kernel_size[0], dtype='float32'),np.arange(kernel_size[1], dtype='float32'))[::-1]
        

        # meshgrids = fluid.layers.meshgrid(
        # # meshgrids = torch.meshgrid(
        #     [   
        #         fluid.layers.arange(start=0, end=size, step=1, dtype='float32')
        #         # torch.arange(size, dtype=torch.float32)
        #         for size in kernel_size
        #         ]
        # )
        
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= np.exp(-(mgrid - mean) ** 2 / (2 * std ** 2 + 10e-9))
            # kernel *= np.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))
            # kernel *= fluid.layers.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))
            # kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        kernel = kernel / np.sum(kernel)
        kernel =np.reshape(kernel, [1, 1, *kernel.shape])
        kernel = kernel.repeat(channels, axis=0)
        
        # Make sure sum of values in gaussian kernel equals 1.
        # kernel = kernel / fluid.layers.sum(kernel)
        # print(kernel)
        # kernel =fluid.layers.reshape(kernel, [1, 1, *kernel.shape])
        # kernel = kernel.numpy().repeat(channels, axis=0)

        # kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        # kernel = kernel.view(1, 1, *kernel.size())
        # kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        
        self.weight=kernel
        # self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = fluid.layers.pad2d(input,  paddings=[self.ka, self.kb, self.ka, self.kb], data_format='NCHW')
        # out = fluid.layers.pad(input,  paddings=[0, 0, 0, 0, self.ka, self.kb, self.ka, self.kb])
        param_attrs = fluid.ParamAttr(learning_rate=1,
                                initializer=fluid.initializer.NumpyArrayInitializer(np.array(self.weight)),
                                trainable=False)
        out = fluid.layers.conv2d(out, num_filters=self.weight.shape[0], filter_size=self.weight.shape[2:], groups=self.groups, param_attr=param_attrs)
        out = fluid.layers.interpolate(out, scale=self.scale)
        # out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        # out = F.conv2d(out, weight=self.weight, groups=self.groups)
        # out = F.interpolate(out, scale_factor=(self.scale, self.scale))
       
        return out
