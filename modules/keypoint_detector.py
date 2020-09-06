# from torch import nn
# import torch
# import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d
import paddle.fluid as fluid
import numpy as np


class KPDetector(fluid.dygraph.Layer):
# class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)
                                   
        self.kp = fluid.dygraph.Conv2D(num_channels=self.predictor.out_filters, num_filters=num_kp, filter_size=[7, 7], padding=pad)
        # self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
        #                     padding=pad)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp

            self.jacobian = fluid.dygraph.Conv2D(num_channels=self.predictor.out_filters,
                                      num_filters=4 * self.num_jacobian_maps, filter_size=[7, 7], padding=pad, param_attr=fluid.initializer.ConstantInitializer(value=0.0), bias_attr=fluid.initializer.NumpyArrayInitializer(np.array([1, 0, 0, 1] * self.num_jacobian_maps)))
            # self.jacobian.weight.data.zero_()
            # self.jacobian.bias.data.copy_(fluid.Tensor([1, 0, 0, 1] * self.num_jacobian_maps))
            # self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
            #                           out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            # self.jacobian.weight.data.zero_()
            # self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = fluid.layers.unsqueeze(heatmap, axes=-1)
        grid = make_coordinate_grid(shape[2:])
        grid = fluid.layers.unsqueeze(grid, axes=0)
        grid = fluid.layers.unsqueeze(grid, axes=0)
        # heatmap = heatmap.unsqueeze(-1)
        # grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        hg=np.sum((heatmap * grid).numpy(), axis=(2, 3))
        value=fluid.dygraph.to_variable(hg)
        # value = fluid.layers.sum(heatmap * grid)
        # value = (heatmap * grid).sum(dim=(2, 3))
        kp = {'value': value}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = fluid.layers.reshape(prediction, (final_shape[0], final_shape[1], -1))
        heatmap = fluid.layers.softmax(heatmap / self.temperature, axis=2)
        heatmap = fluid.layers.reshape(heatmap, final_shape)
        # heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        # heatmap = F.softmax(heatmap / self.temperature, dim=2)
        # heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            jacobian_map = fluid.layers.reshape(jacobian_map, (final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3]))
            heatmap = fluid.layers.unsqueeze(heatmap, axes=2)
            # jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
            #                                     final_shape[3])
            # heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map

            jacobian = fluid.layers.reshape(jacobian, (final_shape[0], final_shape[1], 4, -1))
            jacobian = fluid.dygraph.to_variable(np.sum(jacobian.numpy(), axis=-1))
            jacobian = fluid.layers.reshape(jacobian, (jacobian.shape[0], jacobian.shape[1], 2, 2))
            # jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            # jacobian = jacobian.sum(dim=-1)
            # jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            out['jacobian'] = jacobian

        return out
