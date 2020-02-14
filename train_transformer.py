import os
import torch
import torch.nn as nn
from models import GPLoss, CPLoss, WingLoss, Estimator, Regressor, Discrim
from utils.dataset import GeneralDataset
from torch.utils.data import ConcatDataset
from utils import *
from utils.args import parse_args
import tqdm
from kornia.color import denormalize, normalize, rgb_to_grayscale

from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)

def train(arg):
    log_writer = None
    if arg.save_logs:
        if not os.path.exists('./logs'):
            os.mkdir('./logs')
        log_writer = SummaryWriter()

    epoch = None
    devices = get_devices_list(arg)

    print('*****  Normal Training  *****')
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

    optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr=arg.lr, momentum=arg.momentum,
                                          weight_decay=arg.weight_decay)

    criterion_gp = GPLoss()
    criterion_cp = CPLoss()

    mean = None
    std = None
    if regressor is not None:
        mean = torch.FloatTensor(means_color[arg.dataset][arg.split])
        std = torch.FloatTensor(stds_color[arg.dataset][arg.split])

        if arg.cuda:
            mean = mean.cuda(device=devices[0])
            std = std.cuda(device=devices[0])

    criterion_regressor = None
    if regressor is not None:
        if arg.loss_type == 'L2':
            criterion_regressor = nn.MSELoss()
        elif arg.loss_type == 'L1':
            criterion_regressor = nn.L1Loss()
        elif arg.loss_type == 'smoothL1':
            criterion_regressor = nn.SmoothL1Loss()
        else:
            criterion_regressor = WingLoss(omega=arg.wingloss_w, epsilon=arg.wingloss_e)
        if arg.cuda:
            criterion_regressor = criterion_regressor.cuda(device=devices[0])

    print('Loading dataset ...')

    trainset_list = []
    for key in dataset_size[arg.dataset]:
        trainset_list.append(GeneralDataset(arg, dataset=arg.dataset, split=key))
    trainset_a = ConcatDataset(trainset_list)
    trainset_b = GeneralDataset(arg, dataset=arg.dataset, split=arg.split)

    print('Loading dataset done!')

    dataloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, shuffle=arg.shuffle,
                                             num_workers=arg.workers, pin_memory=True)

    # evolving training
    print('Start training ...')
    for epoch in range(arg.resume_epoch, arg.max_epoch):
        forward_times_per_epoch, sum_loss_decoder = 0, 0.

        if epoch in arg.step_values:
            optimizer_decoder.param_groups[0]['lr'] *= arg.gamma

        for data in tqdm.tqdm(dataloader):
            forward_times_per_epoch += 1
            _, input_images, input_images_norm, gt_coords_xy, _, _, _, _ = data

            if arg.cuda:
                input_images = input_images.cuda(device=devices[0])
                input_images_norm = input_images_norm.cuda(device=devices[0])
                gt_coords_xy = gt_coords_xy.cuda(device=devices[0])

            optimizer_decoder.zero_grad()
            heatmaps_orig = estimator(input_images)[-1]
            heatmaps = F.interpolate(heatmaps_orig, 256, mode='bicubic')
            fake_images_norm = decoder(heatmaps)
            loss_gp = criterion_gp(fake_images_norm, input_images_norm)
            loss_cp = criterion_cp(fake_images_norm, input_images_norm)
            loss_decoder = arg.gp_loss_lambda * loss_gp + arg.cp_loss_lambda * loss_cp

            if regressor is not None:
                fake_images = denormalize(fake_images_norm, mean, std)
                fake_images = rgb_to_grayscale(fake_images)
                fake_images = normalize(fake_images, torch.mean(fake_images), torch.std(fake_images))
                regressor_out = regressor(fake_images, heatmaps_orig)
                loss_regressor = criterion_regressor(regressor_out, gt_coords_xy)
                loss_decoder = loss_decoder + arg.regressor_loss_lambda * loss_regressor

            loss_decoder.backward()
            optimizer_decoder.step()

            sum_loss_decoder += loss_decoder.item()

        if (epoch+1) % arg.save_interval == 0:
            torch.save(decoder.state_dict(), arg.save_folder + 'decoder_' + arg.dataset + '_' + arg.split + '_' + str(epoch+1) + '.pth')

        # if log_writer is not None:
        #     log_writer.add_scalar()

        print('\nepoch: {:0>4d} | loss_decoder: {:.2f}'.format(
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