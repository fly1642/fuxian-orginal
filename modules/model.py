# from torch import nn
# import torch
# import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
# from torchvision import models
import numpy as np
# from torch.autograd import grad
import paddle.fluid as fluid


# class Vgg19(fluid.dygraph.Layer):
# # class Vgg19(torch.nn.Module):
#     """
#     Vgg19 network for perceptual loss. See Sec 3.3.
#     """
#     def __init__(self, requires_grad=False):
#         super(Vgg19, self).__init__()
#         vgg_pretrained_features = 0
        
#         self.slice1 = fluid.dygraph.Sequential()
#         self.slice2 = fluid.dygraph.Sequential()
#         self.slice3 = fluid.dygraph.Sequential()
#         self.slice4 = fluid.dygraph.Sequential()
#         self.slice5 = fluid.dygraph.Sequential()
    
#         for x in range(2):
#             self.slice1.add_sublayer(str(x), vgg_pretrained_features[x])
#         for x in range(2, 7):
#             self.slice2.add_sublayer(str(x), vgg_pretrained_features[x])
#         for x in range(7, 12):
#             self.slice3.add_sublayer(str(x), vgg_pretrained_features[x])
#         for x in range(12, 21):
#             self.slice4.add_sublayer(str(x), vgg_pretrained_features[x])
#         for x in range(21, 30):
#             self.slice5.add_sublayer(str(x), vgg_pretrained_features[x])

#         # vgg_pretrained_features = models.vgg19(pretrained=True).features
#         # self.slice1 = torch.nn.Sequential()
#         # self.slice2 = torch.nn.Sequential()
#         # self.slice3 = torch.nn.Sequential()
#         # self.slice4 = torch.nn.Sequential()
#         # self.slice5 = torch.nn.Sequential()
#         # for x in range(2):
#         #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         # for x in range(2, 7):
#         #     self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         # for x in range(7, 12):
#         #     self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         # for x in range(12, 21):
#         #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         # for x in range(21, 30):
#         #     self.slice5.add_module(str(x), vgg_pretrained_features[x])

#         self.mean = fluid.dygraph.ParameterList(parameters=np.array([0.485, 0.456, 0.406].reshape((1, 3, 1, 1))))
#         self.std = fluid.dygraph.ParameterList(parameters=np.array([0.229, 0.224, 0.225].reshape((1, 3, 1, 1))))
#         # self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
#         #                                requires_grad=False)
#         # self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
#         #                               requires_grad=False)

#         if not requires_grad:
#             for param in self.parameters():
#                 param.stop_gradient = True
#         # if not requires_grad:
#         #     for param in self.parameters():
#         #         param.requires_grad = False

#     def forward(self, X):
#         X = (X - self.mean) / self.std
#         h_relu1 = self.slice1(X)
#         h_relu2 = self.slice2(h_relu1)
#         h_relu3 = self.slice3(h_relu2)
#         h_relu4 = self.slice4(h_relu3)
#         h_relu5 = self.slice5(h_relu4)
#         out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
#         return out


class ConvBlock(fluid.dygraph.Layer):
    """
    卷积+池化
    """
    def __init__(self, name_scope, num_channels, num_filters, groups):
        """构造函数"""
        super(ConvBlock, self).__init__(name_scope)

        self._conv2d_list = []
        init_num_channels = num_channels
        for i in range(groups):
            conv2d = self.add_sublayer(
                'bb_%d' % i,
                fluid.dygraph.Conv2D(
                    init_num_channels, num_filters=num_filters, filter_size=3,
                    stride=1, padding=1, act='relu'
                )
            )
            self._conv2d_list.append(conv2d)
            init_num_channels = num_filters

        self._pool = fluid.dygraph.Pool2D(
            pool_size=2, pool_type='max', pool_stride=2
        )

    def forward(self, inputs):
        """前向计算"""
        x = inputs
        for conv in self._conv2d_list:
            x = conv(x)
        x = self._pool(x)
        return x
class Vgg19(fluid.dygraph.Layer):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        
        nums=[2, 2, 4, 4, 4]
        self.slice1 = ConvBlock(self.full_name(), num_channels=3, num_filters=64, groups=nums[0])
        self.slice2 = ConvBlock(self.full_name(), num_channels=64, num_filters=128, groups=nums[1])
        self.slice3 = ConvBlock(self.full_name(), num_channels=128, num_filters=256, groups=nums[2])
        self.slice4 = ConvBlock(self.full_name(), num_channels=256, num_filters=512, groups=nums[3])
        self.slice5 = ConvBlock(self.full_name(), num_channels=512, num_filters=512, groups=nums[4])

        self.mean = fluid.dygraph.to_variable(np.array([0.485, 0.456, 0.406], dtype='f4').reshape((1, 3, 1, 1)))
        self.std = fluid.dygraph.to_variable(np.array([0.229, 0.224, 0.225], dtype='f4').reshape((1, 3, 1, 1)))


    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class ImagePyramide(fluid.dygraph.Layer):
# class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = downs
        # self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = fluid.layers.gaussian_random(shape=[bs, 2, 3], mean=0, std=kwargs['sigma_affine'])
        self.theta = noise + fluid.layers.reshape(fluid.layers.eye(2, 3), [1, 2, 3])
        # noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        # self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']))
            self.control_points = fluid.layers.unsqueeze(self.control_points, 0)
            self.control_params = fluid.layers.gaussian_random(shape=[bs, 1, kwargs['points_tps'] ** 2], mean=0,
                                               std=kwargs['sigma_tps'])
            # self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            # self.control_points = self.control_points.unsqueeze(0)
            # self.control_params = torch.normal(mean=0,
            #                                    std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = fluid.layers.unsqueeze(make_coordinate_grid(frame.shape[2:]), 0)
        grid = fluid.layers.reshape(grid, [1, frame.shape[2] * frame.shape[3], 2])
        grid = fluid.layers.reshape(self.warp_coordinates(grid), [self.bs, frame.shape[2], frame.shape[3], 2])
        return fluid.layers.grid_sampler(frame, grid)
        # grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        # grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        # grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        # return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta
        theta = fluid.layers.unsqueeze(theta, 1)
        a=theta[:, :, :, :2].numpy()
        b=fluid.layers.unsqueeze(coordinates, -1).numpy()
        transformed = fluid.dygraph.to_variable(np.matmul(a,b)) + theta[:, :, :, 2:]
        transformed = fluid.layers.squeeze(transformed, [-1])
        # theta = self.theta.type(coordinates.type())
        # theta = theta.unsqueeze(1)
        # transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        # transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points
            control_params = self.control_params
            distances =  fluid.layers.reshape(coordinates, [coordinates.shape[0], -1, 1, 2]) -  fluid.layers.reshape(control_points, [1, 1, -1, 2])
            distances = fluid.layers.abs(distances)
            distances = fluid.dygraph.to_variable(np.sum(distances.numpy(), axis=-1))
            # control_points = self.control_points.type(coordinates.type())
            # control_params = self.control_params.type(coordinates.type())
            # distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            # distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * fluid.layers.log(distances + 1e-6)
            result = result * control_params
            result =  fluid.layers.reshape(fluid.dygraph.to_variable(np.sum(result.numpy(), axis=2)), [self.bs, coordinates.shape[1], 1])
            # result = result * torch.log(distances + 1e-6)
            # result = result * control_params
            # result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        na=fluid.dygraph.to_variable(np.sum(new_coordinates[..., 0].numpy()))
        nb=fluid.dygraph.to_variable(np.sum(new_coordinates[..., 1].numpy()))
        grad_x = grad(na, coordinates, create_graph=True)
        grad_y = grad(nb, coordinates, create_graph=True)
        jacobian = fluid.layers.concat([fluid.layers.unsqueeze(grad_x[0], -2), fluid.layers.unsqueeze(grad_y[0], -2)], axis=-2)
        # grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        # grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        # jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}

class GeneratorFullModel(fluid.dygraph.Layer):
# class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        # if torch.cuda.is_available():
        #     self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            # if torch.cuda.is_available():
            #     self.vgg = self.vgg.cuda()

    def forward(self, x):
        x['source']=fluid.dygraph.to_variable(x['source'])
        x['driving']=fluid.dygraph.to_variable(x['driving'])

        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])

        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = fluid.layers.reduce_mean(fluid.layers.abs(x_vgg[i] - y_vgg[i].detach()))
                    # value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = fluid.layers.reduce_mean(((1 - discriminator_maps_generated[key]) ** 2))
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = fluid.layers.reduce_mean(fluid.layers.abs(a - b))
                        # value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = fluid.layers.reduce_mean(fluid.layers.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])))
                # value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            # if self.loss_weights['equivariance_jacobian'] != 0:
            if self.loss_weights['equivariance_jacobian'] == 0:
                jacobian_transformed = fluid.layers.matmul(transform.jacobian(transformed_kp['value']),
                                                    transformed_kp['jacobian'])
                normed_driving = fluid.dygraph.to_variable(np.linalg.inv(kp_driving['jacobian'].numpy()))
                normed_transformed = jacobian_transformed
                value = fluid.layers.matmul(normed_driving, normed_transformed)
                eye = fluid.layers.reshape(fluid.layers.eye(2), shape=[1, 1, 2, 2])
                value = fluid.layers.reduce_mean(fluid.layers.abs(eye - value))
                # jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                #                                     transformed_kp['jacobian'])

                # normed_driving = torch.inverse(kp_driving['jacobian'])
                # normed_transformed = jacobian_transformed
                # value = torch.matmul(normed_driving, normed_transformed)

                # eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                # value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        return loss_values, generated

class DiscriminatorFullModel(fluid.dygraph.Layer):
# class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        # if torch.cuda.is_available():
        #     self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        x['driving']=fluid.dygraph.to_variable(x['driving'])

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * fluid.layers.reduce_mean(value)
            # value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values
