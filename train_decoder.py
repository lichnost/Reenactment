import os

import torch.nn as nn
import tqdm
from kornia.color import denormalize, normalize, rgb_to_grayscale, rgb_to_bgr
from kornia import tensor_to_image
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

    decoder = create_model_decoder(arg, devices)
    decoder.train()

    discrim = None
    if arg.GAN:
        discrim = create_model_decoder_discrim(arg, devices)
        discrim.train()

    edge = create_model_edge(arg, devices, eval=True)
    edge.eval()

    print('Creating networks done!')

    optimizer_decoder, scheduler_decoder = create_optimizer(arg, decoder.parameters())

    optimizer_discrim = None
    scheduler_discrim = None
    if discrim:
        optimizer_discrim, scheduler_discrim = create_optimizer(arg, discrim.parameters())

    criterion_gp = None
    if arg.gp_loss:
        criterion_gp = GPLoss()
        if arg.cuda:
            criterion_gp = criterion_gp.cuda(device=devices[0])

    criterion_cp = None
    if arg.cp_loss:
        criterion_cp = CPLoss()
        if arg.cuda:
            criterion_cp = criterion_cp.cuda(device=devices[0])

    criterion_pixel = nn.L1Loss()
    if arg.cuda:
        criterion_pixel = criterion_pixel.cuda(device=devices[0])

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

    criterion_gan = None
    if discrim is not None:
        criterion_gan = nn.BCELoss()

        discrim_false = torch.zeros((arg.batch_size, 1, 30, 30))
        discrim_true = torch.ones((arg.batch_size, 1, 30, 30))
        if arg.cuda:
            criterion_gan = criterion_gan.cuda(device=devices[0])

            discrim_false = discrim_false.cuda(device=devices[0])
            discrim_true = discrim_true.cuda(device=devices[0])



    print('Loading dataset ...')
    trainset = DecoderDataset(arg, dataset=arg.dataset, split=arg.split)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, shuffle=arg.shuffle,
                                             num_workers=arg.workers, pin_memory=True)
    steps_per_epoch = len(dataloader)

    mean = torch.FloatTensor(means_color[arg.dataset][arg.split])
    std = torch.FloatTensor(stds_color[arg.dataset][arg.split])
    if arg.cuda:
        mean = mean.cuda(device=devices[0])
        std = std.cuda(device=devices[0])
    print('Loading dataset done!')

    # evolving training
    print('Start training ...')
    for epoch in range(arg.resume_epoch, arg.max_epoch):
        global_step_base = epoch * steps_per_epoch
        forward_times_per_epoch, sum_loss_decoder, sum_loss_discrim = 0, 0., 0.

        for data in tqdm.tqdm(dataloader):
            forward_times_per_epoch += 1
            global_step = global_step_base + forward_times_per_epoch

            input_images, input_images_norm, gt_coords_xy = data
            input_batch_size = input_images.shape[0]

            # show_img(tensor_to_image(rgb_to_bgr(denormalize(input_images_norm[0].unsqueeze(0), mean, std)).squeeze()), 'target', wait=0, keep=True)

            if arg.cuda:
                input_images = input_images.cuda(device=devices[0])
                input_images_norm = input_images_norm.cuda(device=devices[0])
                gt_coords_xy = gt_coords_xy.cuda(device=devices[0])

            with torch.no_grad():
                heatmaps_orig = estimator(input_images)[-1]
                heatmaps = heatmaps_orig
                heatmaps = edge(heatmaps)

            fake_images_norm = decoder(heatmaps)

            if discrim is not None:
                discrim.train()
                optimizer_discrim.zero_grad()
                heatmaps_discrim = F.interpolate(heatmaps, 256, mode='bilinear')

                # heatmap_show = get_heatmap_gray(heatmaps_discrim[0].unsqueeze(0)).detach().cpu().numpy()
                # heatmap_show = (
                #         255 - np.uint8(255 * (heatmap_show - np.min(heatmap_show)) / np.ptp(heatmap_show)))
                # heatmap_show = np.moveaxis(heatmap_show, 0, -1)
                # show_img(heatmap_show, 'heatmap', wait=2000, keep=False)

                discrim_input = torch.cat((heatmaps_discrim, input_images_norm), 1)
                discrim_output = discrim(discrim_input)
                loss_discrim_true = criterion_gan(discrim_output, discrim_true[:input_batch_size, ...])

                discrim_input = torch.cat((heatmaps_discrim, fake_images_norm), 1)
                discrim_output = discrim(discrim_input)
                loss_discrim_fake = criterion_gan(discrim_output, discrim_false[:input_batch_size, ...])
                loss_discrim = (loss_discrim_fake + loss_discrim_true) * 0.5

                log('loss_discrim', loss_discrim.item(), global_step)
                loss_discrim.backward(retain_graph=True)
                optimizer_discrim.step()

            optimizer_decoder.zero_grad()

            loss_decoder = None
            if criterion_gp is not None:
                loss_gp = criterion_gp(fake_images_norm, input_images_norm)
                loss_decoder = loss_decoder + arg.loss_gp_lambda * loss_gp if loss_decoder is not None else arg.loss_gp_lambda * loss_gp
                log('loss_gp', loss_gp.item(), global_step)

            if criterion_cp is not None:
                loss_cp = criterion_cp(fake_images_norm, input_images_norm)
                loss_decoder = loss_decoder + arg.loss_cp_lambda * loss_cp if loss_decoder is not None else arg.loss_cp_lambda * loss_cp
                log('loss_cp', loss_cp.item(), global_step)

            loss_pixel = criterion_pixel(fake_images_norm, input_images_norm)
            loss_decoder = loss_decoder + arg.loss_pixel_lambda * loss_pixel if loss_decoder is not None else arg.loss_pixel_lambda * loss_pixel
            log('loss_pixel', loss_pixel.item(), global_step)

            if criterion_feature is not None:
                loss_feature = criterion_feature(fake_images_norm, input_images_norm)
                loss_decoder = loss_decoder + arg.loss_feature_lambda * loss_feature

                log('loss_feature', loss_feature.item(), global_step)

            if regressor is not None:
                with torch.no_grad():
                    fake_images = denormalize(fake_images_norm, mean, std)
                    fake_images = rgb_to_grayscale(fake_images)
                    fake_images = normalize(fake_images, torch.mean(fake_images), torch.std(fake_images))

                    regressor_out = regressor(fake_images, heatmaps_orig)

                loss_regressor = criterion_regressor(regressor_out, gt_coords_xy)
                loss_decoder = loss_decoder + arg.loss_regressor_lambda * loss_regressor

                log('loss_regressor', loss_regressor.item(), global_step)

            if discrim is not None:
                discrim.eval()
                with torch.no_grad():
                    discrim_output = discrim(discrim_input)

                loss_gan = criterion_gan(discrim_output, discrim_true[:input_batch_size, ...])
                loss_decoder = loss_decoder + arg.loss_discrim_lambda * loss_gan
                log('loss_gan', loss_gan.item(), global_step)

            log('loss_decoder', loss_decoder.item(), global_step)

            loss_decoder.backward()
            optimizer_decoder.step()

            sum_loss_decoder += loss_decoder.item()
            if discrim is not None:
                sum_loss_discrim += loss_discrim.item()

            if arg.save_logs:
                images_to_save = np.uint8(denormalize(fake_images_norm[0, ...], mean, std).detach().cpu().numpy())
                log_img('fake_image', images_to_save, global_step)

        mean_loss_decoder = sum_loss_decoder / forward_times_per_epoch
        mean_loss_discrim = sum_loss_discrim / forward_times_per_epoch

        scheduler_decoder.step(mean_loss_decoder)
        if scheduler_discrim is not None:
            scheduler_discrim.step(mean_loss_discrim)

        if (epoch+1) % arg.save_interval == 0:
            torch.save(decoder.state_dict(), arg.save_folder + 'decoder_' + arg.dataset + '_' + arg.split + '_' + str(epoch+1) + '.pth')
            torch.save(discrim.state_dict(), arg.save_folder + 'decoder_discrim_' + arg.dataset + '_' + arg.split + '_' + str(
                           epoch + 1) + '.pth')

        print('\nepoch: {:0>4d} | loss_decoder: {:.10f}'.format(
            epoch,
            mean_loss_decoder,
        ))

    torch.save(decoder.state_dict(), arg.save_folder + 'decoder_' + arg.dataset + '_' + arg.split + '_' + str(epoch+1) + '.pth')
    torch.save(discrim.state_dict(), arg.save_folder + 'decoder_discrim_' + arg.dataset + '_' + arg.split + '_' + str(epoch + 1) + '.pth')
    print('Training done!')


if __name__ == '__main__':
    arg = parse_args()

    if not os.path.exists(arg.save_folder):
        os.mkdir(arg.save_folder)
    if not os.path.exists(arg.resume_folder):
        os.mkdir(arg.resume_folder)

    train(arg)