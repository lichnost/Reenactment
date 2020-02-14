import os

import torch.nn as nn
import tqdm
from kornia.color import denormalize, normalize, rgb_to_grayscale
from torch.utils.tensorboard import SummaryWriter

from models import GPLoss, CPLoss, WingLoss, AdaptiveWingLoss, FeatureLoss
from utils import *
from utils.args import parse_args
from utils.dataset import DecoderDataset


def train(arg):
    log_writer = None
    if arg.save_logs:
        log_path = './logs/decoder_' + arg.dataset + '_' + arg.split
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_writer = SummaryWriter(log_dir=log_path)
        def log(tag, scalar, step):
            log_writer.add_scalar(tag, scalar, step)
    else:
        def log(tag, scalar, step):
            pass

    epoch = None
    devices = get_devices_list(arg)

    print('*****  Training decoder  *****')
    print('Training parameters:\n' +
          '# Dataset:            ' + arg.dataset + '\n' +
          '# Dataset split:      ' + arg.split + '\n' +
          '# Batchsize:          ' + str(arg.batch_size) + '\n' +
          '# Num workers:        ' + str(arg.workers) + '\n' +
          '# PDB:                ' + str(arg.PDB) + '\n' +
          '# Use GPU:            ' + str(arg.cuda) + '\n' +
          '# Start lr:           ' + str(arg.lr) + '\n' +
          '# Max epoch:          ' + str(arg.max_epoch) + '\n' +
          '# Loss type:          ' + arg.loss_type + '\n' +
          '# Heatmap loss type:  ' + arg.gp_loss_type + '\n' +
          '# Resumed model:      ' + str(arg.resume_epoch > 0))
    if arg.resume_epoch > 0:
        print('# Resumed epoch:      ' + str(arg.resume_epoch))

    print('Creating networks ...')
    estimator = create_model_estimator(arg, devices, eval=True)
    estimator.eval()

    regressor = None
    if arg.regressor_loss:
        regressor = create_model_regressor(arg, devices, eval=True)
        regressor.eval()

    decoder = create_model_decoder(arg, devices)
    decoder.train()

    print('Creating networks done!')

    optimizer_decoder = create_optimizer(arg, decoder.parameters())

    criterion_gp = GPLoss()
    criterion_cp = CPLoss()
    if arg.cuda:
        criterion_gp = criterion_gp.cuda(device=devices[0])
        criterion_cp = criterion_cp.cuda(device=devices[0])

    mean = None
    std = None
    if regressor is not None:
        mean = torch.FloatTensor(means_color[arg.dataset][arg.split])
        std = torch.FloatTensor(stds_color[arg.dataset][arg.split])

        if arg.cuda:
            mean = mean.cuda(device=devices[0])
            std = std.cuda(device=devices[0])

    criterion_simple = nn.MSELoss()
    if arg.cuda:
        criterion_simple = criterion_simple.cuda(device=devices[0])

    criterion_feature = None
    if arg.feature_loss:
        criterion_feature = FeatureLoss(False, arg.feature_loss_type)
        if arg.cuda:
            criterion_feature = criterion_feature.cuda(device=devices[0])

    criterion_regressor = None
    if regressor is not None:
        if arg.loss_type == 'L2':
            criterion_regressor = nn.MSELoss()
        elif arg.loss_type == 'L1':
            criterion_regressor = nn.L1Loss()
        elif arg.loss_type == 'smoothL1':
            criterion_regressor = nn.SmoothL1Loss()
        elif arg.loss_type == 'wingloss':
            criterion_regressor = WingLoss(omega=arg.wingloss_omega, epsilon=arg.wingloss_epsilon)
        else:
            criterion_regressor = AdaptiveWingLoss(arg.wingloss_omega, theta=arg.wingloss_theta, epsilon=arg.wingloss_epsilon,
                                         alpha=arg.wingloss_alpha)
        if arg.cuda:
            criterion_regressor = criterion_regressor.cuda(device=devices[0])


    print('Loading dataset ...')
    trainset = DecoderDataset(arg, dataset=arg.dataset, split=arg.split)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, shuffle=arg.shuffle,
                                             num_workers=arg.workers, pin_memory=True)
    steps_per_epoch = len(dataloader)
    print('Loading dataset done!')



    # evolving training
    print('Start training ...')
    for epoch in range(arg.resume_epoch, arg.max_epoch):
        global_step_base = epoch * steps_per_epoch
        forward_times_per_epoch, sum_loss_decoder = 0, 0.

        if epoch in arg.step_values:
            optimizer_decoder.param_groups[0]['lr'] *= arg.gamma

        for data in tqdm.tqdm(dataloader):
            forward_times_per_epoch += 1
            global_step = global_step_base + forward_times_per_epoch

            input_images, input_images_norm, gt_coords_xy = data

            if arg.cuda:
                input_images = input_images.cuda(device=devices[0])
                input_images_norm = input_images_norm.cuda(device=devices[0])
                gt_coords_xy = gt_coords_xy.cuda(device=devices[0])

            optimizer_decoder.zero_grad()
            with torch.no_grad():
                heatmaps_orig = estimator(input_images)[-1]
                heatmaps = F.interpolate(heatmaps_orig, 256, mode='bicubic')
                heatmaps[heatmaps < arg.boundary_cutoff_lambda * heatmaps.max()] = 0

            fake_images_norm = decoder(heatmaps)
            loss_gp = criterion_gp(fake_images_norm, input_images_norm)
            loss_cp = criterion_cp(fake_images_norm, input_images_norm)
            loss_simple = criterion_simple(fake_images_norm, input_images_norm)
            loss_decoder = arg.gp_loss_lambda * loss_gp + arg.cp_loss_lambda * loss_cp + loss_simple

            log('loss_gp', loss_gp.item(), global_step)
            log('loss_cp', loss_cp.item(), global_step)
            log('loss_simple', loss_simple.item(), global_step)

            if criterion_feature is not None:
                loss_feature = criterion_feature(fake_images_norm, input_images_norm)
                loss_decoder = loss_decoder + arg.feature_loss_lambda * loss_feature

                log('loss_feature', loss_feature.item(), global_step)

            if regressor is not None:
                fake_images = denormalize(fake_images_norm, mean, std)
                fake_images = rgb_to_grayscale(fake_images)
                fake_images = normalize(fake_images, torch.mean(fake_images), torch.std(fake_images))

                with torch.no_grad():
                    regressor_out = regressor(fake_images, heatmaps_orig)

                loss_regressor = criterion_regressor(regressor_out, gt_coords_xy)
                loss_decoder = loss_decoder + arg.regressor_loss_lambda * loss_regressor

                log('loss_regressor', loss_regressor.item(), global_step)

            log('loss_decoder', loss_decoder.item(), global_step)

            loss_decoder.backward()
            optimizer_decoder.step()

            sum_loss_decoder += loss_decoder.item()

        if (epoch+1) % arg.save_interval == 0:
            torch.save(decoder.state_dict(), arg.save_folder + 'decoder_' + arg.dataset + '_' + arg.split + '_' + str(epoch+1) + '.pth')

        # if log_writer is not None:
        #     log_writer.add_scalar()

        print('\nepoch: {:0>4d} | loss_decoder: {:.10f}'.format(
            epoch,
            sum_loss_decoder/forward_times_per_epoch,
        ))

    torch.save(decoder.state_dict(), arg.save_folder + 'decoder_' + arg.dataset + '_' + arg.split + '_' + str(epoch+1) + '.pth')
    print('Training done!')


if __name__ == '__main__':
    arg = parse_args()

    if not os.path.exists(arg.save_folder):
        os.mkdir(arg.save_folder)
    if not os.path.exists(arg.resume_folder):
        os.mkdir(arg.resume_folder)

    train(arg)