from tqdm import trange
# import torch
# from torch.utils.data import DataLoader
import paddle.fluid as fluid
import numpy as np

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

# from torch.optim.lr_scheduler import MultiStepLR

# from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater


def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    boundaries = train_params['epoch_milestones']
    values = [train_params['lr_generator']]
    for i in range(len(boundaries)):
        values.append(0.1*values[-1])
    optimizer_generator = fluid.optimizer.AdamOptimizer(parameter_list=generator.parameters(), learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0), beta1=0.5, beta2=0.999)
    optimizer_discriminator = fluid.optimizer.AdamOptimizer(parameter_list=discriminator.parameters(), learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0), beta1=0.5, beta2=0.999)
    optimizer_kp_detector = fluid.optimizer.AdamOptimizer(parameter_list=kp_detector.parameters(), learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0), beta1=0.5, beta2=0.999)
    # optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    # optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    # optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0

    # scheduler_generator = fluid.dygraph.StepDecay(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
    #                                   last_epoch=start_epoch - 1)
    # scheduler_discriminator = fluid.dygraph.StepDecay(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
    #                                       last_epoch=start_epoch - 1)
    # scheduler_kp_detector = fluid.dygraph.StepDecay(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
    #                                     last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    # scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
    #                                   last_epoch=start_epoch - 1)
    # scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
    #                                       last_epoch=start_epoch - 1)
    # scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
    #                                     last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = fluid.io.batch(dataset.reader, train_params['batch_size'], drop_last=True)
    dataloader = fluid.io.shuffle(dataloader, 3)
    # dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    # if torch.cuda.is_available():
    #     generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
    #     discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            batch_num=0
            for batch_x in dataloader():
                batch_num+=1
                x={'driving':[],'source':[],'name':[]}
                for xi in batch_x:
                    x['driving'].append(xi['driving'])
                    x['source'].append(xi['source'])
                    x['name'].append(xi['name'])
                for arr in x:
                    x[arr]=np.array(x[arr])
                losses_generator, generated = generator_full(x)

                loss_values = [fluid.layers.reduce_mean(val).numpy() for val in losses_generator.values()]
                G_loss = fluid.dygraph.to_variable(np.array([np.sum(loss_values)]))
                # loss_values = [val.mean() for val in losses_generator.values()]
                # loss = sum(loss_values)

                G_loss.backward()
                optimizer_generator.minimize(G_loss)
                generator.clear_gradients()
                optimizer_kp_detector.minimize(G_loss)
                kp_detector.clear_gradients()
                # optimizer_generator.step()
                # optimizer_generator.zero_grad()
                # optimizer_kp_detector.step()
                # optimizer_kp_detector.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.clear_gradients()
                    # optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [fluid.layers.reduce_mean(val).numpy() for val in losses_discriminator.values()]
                    D_loss = fluid.dygraph.to_variable(np.array([np.sum(loss_values)]))
                    # loss_values = [val.mean() for val in losses_discriminator.values()]
                    # loss = sum(loss_values)
                    D_loss.backward()
                    optimizer_discriminator.minimize(D_loss)
                    discriminator.clear_gradients()
                    # optimizer_discriminator.step()
                    # optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: fluid.layers.reduce_mean(value).detach().numpy() for key, value in losses_generator.items()}
                # losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

                # 打印输出
                if(batch_num % 400 == 0):
                    print('epoch =', epoch, ', batch =', batch_num, ', generator_loss =', G_loss.numpy(), 'discriminator_loss =', D_loss.numpy())
                
            # 存储模型
            fluid.save_dygraph(generator.state_dict(), log_dir+'g')
            fluid.save_dygraph(optimizer_generator.state_dict(), log_dir+'g')
            # fluid.save_dygraph(discriminator.state_dict(), log_dir+'d')
            fluid.save_dygraph(optimizer_discriminator.state_dict(), log_dir+'d')
            fluid.save_dygraph(kp_detector.state_dict(), log_dir+'kp')
            fluid.save_dygraph(optimizer_kp_detector.state_dict(), log_dir+'kp')
            
            # scheduler_generator.step()
            # scheduler_discriminator.step()
            # scheduler_kp_detector.step()
            
            # logger.log_epoch(epoch, {'generator': generator,
            #                          'discriminator': discriminator,
            #                          'kp_detector': kp_detector,
            #                          'optimizer_generator': optimizer_generator,
            #                          'optimizer_discriminator': optimizer_discriminator,
            #                          'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated)
