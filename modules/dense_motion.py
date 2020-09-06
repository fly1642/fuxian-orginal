# from torch import nn
# import torch.nn.functional as F
# import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
import paddle.fluid as fluid
import numpy as np

class DenseMotionNetwork(fluid.dygraph.Layer):
# class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 1) * (num_channels + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = fluid.dygraph.Conv2D(num_channels=self.hourglass.out_filters, num_filters=num_kp + 1, filter_size=[7, 7], padding=[3, 3])
        # self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=(7, 7), padding=(3, 3))

        if estimate_occlusion_map:
            self.occlusion = fluid.dygraph.Conv2D(num_channels=self.hourglass.out_filters, num_filters=1, filter_size=[7, 7], padding=[3, 3])
            # self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        #adding background feature
        zeros = fluid.layers.zeros([heatmap.shape[0], 1, spatial_size[0], spatial_size[1]], dtype='float32')
        heatmap = fluid.layers.concat([zeros, heatmap], axis=1)
        heatmap = fluid.layers.unsqueeze(heatmap, axes=2)
        # zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        # heatmap = torch.cat([zeros, heatmap], dim=1)
        # heatmap = heatmap.unsqueeze(2)
        return heatmap

    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w))
        identity_grid = fluid.layers.reshape(identity_grid, [1, 1, h, w, 2])
        coordinate_grid = identity_grid - fluid.layers.reshape(kp_driving['value'], (bs, self.num_kp, 1, 1, 2))
        # identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
        # identity_grid = identity_grid.view(1, 1, h, w, 2)
        # coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)
        if 'jacobian' in kp_driving:
            jacobian = fluid.layers.matmul(kp_source['jacobian'], fluid.dygraph.to_variable(np.linalg.inv(kp_driving['jacobian'].numpy())))
            jacobian = fluid.layers.unsqueeze(jacobian, axes=-3)
            jacobian = fluid.layers.unsqueeze(jacobian, axes=-3)
            jacobian = jacobian.numpy()
            repeats = (1, 1, h, w, 1, 1)
            for i in range(len(repeats)):
                jacobian =np.repeat(jacobian, repeats[i],axis=i)
            jacobian=fluid.dygraph.to_variable(jacobian)
            coordinate_grid = fluid.layers.matmul(jacobian, fluid.layers.unsqueeze(coordinate_grid, axes=-1))
            coordinate_grid = fluid.layers.squeeze(coordinate_grid, axes=[-1])
            # jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            # jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
            # jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
            # coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            # coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + fluid.layers.reshape(kp_source['value'], (bs, self.num_kp, 1, 1, 2))
        # driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 2)

        #adding background feature
        identity_grid = identity_grid.numpy()
        repeats = (bs, 1, 1, 1, 1)
        for i in range(len(repeats)):
            identity_grid =np.repeat(identity_grid, repeats[i],axis=i)
        identity_grid=fluid.dygraph.to_variable(identity_grid)
        sparse_motions = fluid.layers.concat([identity_grid, driving_to_source], axis=1)
        # identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        # sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        source_repeat = fluid.layers.unsqueeze(source_image, axes=1)
        source_repeat = fluid.layers.unsqueeze(source_repeat, axes=1)
        repeats = (1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.numpy()
        for i in range(len(repeats)):
            source_repeat =np.repeat(source_repeat, repeats[i], axis=i)
        source_repeat=fluid.dygraph.to_variable(source_repeat)
        # source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)

        source_repeat = fluid.layers.reshape(source_repeat, [bs * (self.num_kp + 1), -1, h, w])
        sparse_motions = fluid.layers.reshape(sparse_motions, [bs * (self.num_kp + 1), h, w, -1])
        sparse_deformed = fluid.layers.grid_sampler(source_repeat, sparse_motions)
        sparse_deformed = fluid.layers.reshape(sparse_deformed, [bs, self.num_kp + 1, -1, h, w])
        # source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        # sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        # sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        # sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict['sparse_deformed'] = deformed_source

        input = fluid.layers.concat([heatmap_representation, deformed_source], axis=2)
        input = fluid.layers.reshape(input, [bs, -1, h, w])
        # input = torch.cat([heatmap_representation, deformed_source], dim=2)
        # input = input.view(bs, -1, h, w)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = fluid.layers.softmax(mask, axis=1)
        out_dict['mask'] = mask
        mask = fluid.layers.unsqueeze(mask, axes=2)
        sparse_motion = fluid.layers.transpose(sparse_motion, perm=[0, 1, 4, 2, 3])
        deformation = fluid.dygraph.to_variable(np.sum((sparse_motion * mask).numpy(), axis=1))
        deformation = fluid.layers.transpose(deformation, perm=[0, 2, 3, 1])
        # mask = F.softmax(mask, dim=1)
        # out_dict['mask'] = mask
        # mask = mask.unsqueeze(2)
        # sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        # deformation = (sparse_motion * mask).sum(dim=1)
        # deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation

        # Sec. 3.2 in the paper
        if self.occlusion:
            occlusion_map = fluid.layers.sigmoid(self.occlusion(prediction))
            # occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
