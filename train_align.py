import os

import torch.nn as nn
import tqdm
from kornia.color import denormalize, normalize, rgb_to_grayscale, rgb_to_bgr
from kornia import tensor_to_image
from torch.utils.tensorboard import SummaryWriter

from models import GPLoss, CPLoss, WingLoss, AdaptiveWingLoss, FeatureLoss
from utils import *
from utils.args import parse_args
from utils.dataset import GeneralDataset


def train(arg):
    log_writer = None
    if arg.save_logs:
        log_path = './logs/align_' + arg.dataset + '_' + arg.split
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_writer = SummaryWriter(log_dir=log_path)
        def log(tag, scalar, step):
            log_writer.add_scalar(tag, scalar, step)
        def log_img(tag, img, step):
            log_writer.add_image(tag, img, step)
    else:
        def log(tag, scalar, step):
            pass
        def log_img(tag, img, step):
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

    align = create_model_align(arg, devices)
    align.train()

    edge = create_model_edge(arg, devices, eval=True)
    edge.eval()

    print('Creating networks done!')

    optimizer_align, scheduler_align = create_optimizer(arg, align.parameters(), create_scheduler=True)

    if arg.loss_type == 'L2':
        criterion = nn.MSELoss()
    elif arg.loss_type == 'L1':
        criterion = nn.L1Loss()
    elif arg.loss_type == 'smoothL1':
        criterion = nn.SmoothL1Loss()
    elif arg.loss_type == 'wingloss':
        criterion = WingLoss(omega=arg.wingloss_omega, epsilon=arg.wingloss_epsilon)
    else:
        criterion = AdaptiveWingLoss(arg.wingloss_omega, theta=arg.wingloss_theta, epsilon=arg.wingloss_epsilon,
                                     alpha=arg.wingloss_alpha)
    if arg.cuda:
        criterion = criterion.cuda(device=devices[0])

    print('Loading dataset ...')
    trainset = GeneralDataset(arg, dataset=arg.dataset, split=arg.split)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, shuffle=arg.shuffle,
                                             num_workers=arg.workers, pin_memory=True)
    steps_per_epoch = len(dataloader)
    print('Loading dataset done!')

    # evolving training
    print('Start training ...')
    for epoch in range(arg.resume_epoch, arg.max_epoch):
        global_step_base = epoch * steps_per_epoch
        forward_times_per_epoch, sum_loss = 0, 0.

        for data in tqdm.tqdm(dataloader):
            forward_times_per_epoch += 1
            global_step = global_step_base + forward_times_per_epoch

            _, input_images, _, gt_coords_xy, _, _, _, _ = data

            if arg.cuda:
                input_images = input_images.cuda(device=devices[0])
                gt_coords_xy = gt_coords_xy.cuda(device=devices[0])

            with torch.no_grad():
                heatmaps = estimator(input_images)[-1]
                # heatmaps[heatmaps < arg.boundary_cutoff_lambda * heatmaps.max()] = 0
                heatmaps = edge(heatmaps)

            optimizer_align.zero_grad()
            coords_predict = align(heatmaps)
            gt_coords_xy = gt_coords_xy/4
            loss = criterion(coords_predict, gt_coords_xy)
            log('loss', loss.item(), global_step)
            loss.backward()
            optimizer_align.step()

            sum_loss += loss.item()

            show_img(tensor_to_image(input_images[0]))

            heatmap_sum = np.uint8(get_heatmap_gray(heatmaps[0].unsqueeze(0), denorm=True).detach().squeeze().cpu().numpy())
            gt_coords_xy = gt_coords_xy[0].detach().cpu().squeeze().numpy()
            for i in range(0, 2*kp_num[arg.dataset], 2):
                draw_circle(heatmap_sum, (int(gt_coords_xy[i]), int(gt_coords_xy[i+1])))
            show_img(cv2.resize(heatmap_sum, (256, 256)))

            heatmap_sum = np.uint8(get_heatmap_gray(heatmaps[0].unsqueeze(0), denorm=True).detach().squeeze().cpu().numpy())
            coords_predict = coords_predict.detach().cpu().squeeze().numpy()
            for i in range(0, 2 * kp_num[arg.dataset], 2):
                draw_circle(heatmap_sum, (int(coords_predict[i]), int(coords_predict[i + 1])))
            show_img(cv2.resize(heatmap_sum, (256, 256)))


        mean_sum_loss = sum_loss / forward_times_per_epoch

        if scheduler_align is not None:
            scheduler_align.step(mean_sum_loss)

        if (epoch+1) % arg.save_interval == 0:
            torch.save(align.state_dict(), arg.save_folder + 'align_' + arg.dataset + '_' + str(epoch+1) + '.pth')

        print('\nepoch: {:0>4d} | loss: {:.10f}'.format(
            epoch,
            mean_sum_loss,
        ))

    torch.save(align.state_dict(), arg.save_folder + 'align_' + arg.dataset + '_' + str(epoch + 1) + '.pth')
    print('Training done!')


if __name__ == '__main__':
    arg = parse_args()

    if not os.path.exists(arg.save_folder):
        os.mkdir(arg.save_folder)
    if not os.path.exists(arg.resume_folder):
        os.mkdir(arg.resume_folder)

    train(arg)