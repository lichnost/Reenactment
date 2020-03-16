import os
import torch
import torch.nn as nn
from utils.dataset import GeneralDataset
from utils import *
from utils.args import parse_args
import tqdm
from kornia.color import denormalize, normalize, rgb_to_grayscale
from models import GPLoss

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import json

def train(arg):
    log_writer = None
    if arg.save_logs:
        log_path = './logs/transformer_' + arg.dataset + '_' + arg.split
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_writer = SummaryWriter(log_dir=log_path)
        def log(tag, scalar, step):
            log_writer.add_scalar(tag, scalar, step)
        def log_img(tag, img, step):
            log_writer.add_image(tag, img, step)
        def log_text(tag, text, step):
            log_writer.add_text(tag, text, step)
    else:
        def log(tag, scalar, step):
            pass
        def log_img(tag, img, step):
            pass
        def log_text(tag, text, step):
            pass

    epoch = None
    devices = get_devices_list(arg)

    print('*****  Normal Training  *****')
    print('Training parameters:\n' +
          '# Dataset source:            ' + arg.dataset + '\n' +
          '# Dataset split source:      ' + arg.split_source + '\n' +
          '# Dataset split target:      ' + arg.split + '\n' +
          '# Batchsize:          ' + str(arg.batch_size) + '\n' +
          '# Num workers:        ' + str(arg.workers) + '\n' +
          '# PDB:                ' + str(arg.PDB) + '\n' +
          '# Use GPU:            ' + str(arg.cuda) + '\n' +
          '# Start lr:           ' + str(arg.lr) + '\n' +
          '# Max epoch:          ' + str(arg.max_epoch) + '\n' +
          '# Resumed model:      ' + str(arg.resume_epoch > 0))
    if arg.resume_epoch > 0:
        print('# Resumed epoch:      ' + str(arg.resume_epoch))

    log_text('arguments', json.dumps(vars(arg), indent=2), 0)

    print('Creating networks ...')
    pca = create_model_pca(arg, devices, eval=True)
    pca.eval()

    estimator = create_model_estimator(arg, devices, eval=True)
    estimator.eval()

    align = create_model_align(arg, devices, eval=True)
    align.eval()

    edge = create_model_edge(arg, devices, eval=True)
    edge.eval()

    # def gen_edge(inputs): # laplas or sobel gradient
    #     #inputs: N*13*64*64  conv: 3*3
    #     size_ = inputs.size()
    #     tmp = [edge(inputs[:, ii, :, :].contiguous().view(size_[0], 1, size_[2], size_[3])) for ii in range(size_[1])]
    #     return torch.cat(tmp, 1)

    generator_a2b = create_model_transformer_a2b(arg, devices, eval=False)
    generator_a2b.train()
    generator_b2a = create_model_transformer_b2a(arg, devices, eval=False)
    generator_b2a.train()

    if arg.GAN:
        discrim_a = create_model_transformer_discrim_a(arg, devices, eval=False)
        discrim_a.train()
        discrim_b = create_model_transformer_discrim_b(arg, devices, eval=False)
        discrim_b.train()

    print('Creating networks done!')

    optimizer_generator, scheduler_generator = create_optimizer(arg, list(generator_a2b.parameters()) + list(generator_b2a.parameters()))

    if arg.GAN:
        lr_restore = arg.lr
        weight_decay_restore = arg.weight_decay
        arg.lr *= 0.5
        arg.weight_decay = 0.1
        optimizer_discrim, scheduler_discrim = create_optimizer(arg, list(discrim_a.parameters()) + list(discrim_b.parameters()))
        arg.lr = lr_restore
        arg.weight_decay = weight_decay_restore

    # mean_target = torch.FloatTensor(means_color[arg.dataset][arg.split])
    # std_target = torch.FloatTensor(stds_color[arg.dataset][arg.split])
    # mean_source = torch.FloatTensor(means_color[arg.dataset][arg.split_source])
    # std_source = torch.FloatTensor(stds_color[arg.dataset][arg.split_source])
    # if arg.cuda:
    #     mean_target = mean_target.cuda(device=devices[0])
    #     std_target = std_target.cuda(device=devices[0])
    #     mean_source = mean_source.cuda(device=devices[0])
    #     std_source = std_source.cuda(device=devices[0])

    if arg.GAN:
        criterion_gan = nn.MSELoss()
        discrim_false = torch.zeros((arg.batch_size, 1, 6, 6))
        discrim_true = torch.ones((arg.batch_size, 1, 6, 6))
        if arg.cuda:
            criterion_gan = criterion_gan.cuda(device=devices[0])

            discrim_false = discrim_false.cuda(device=devices[0])
            discrim_true = discrim_true.cuda(device=devices[0])

    criterion_pixel = nn.L1Loss()
    if arg.cuda:
        criterion_pixel = criterion_pixel.cuda(device=devices[0])

    criterion_gp = None
    if arg.gp_loss:
        criterion_gp = GPLoss()
        if arg.cuda:
            criterion_gp = criterion_gp.cuda(device=devices[0])

    criterion_pca = nn.L1Loss()
    if arg.cuda:
        criterion_pca = criterion_pca.cuda(device=devices[0])

    print('Loading dataset ...')

    trainset_a = GeneralDataset(arg, dataset=arg.dataset_source, split=arg.split_source)
    trainset_b = GeneralDataset(arg, dataset=arg.dataset, split=arg.split)

    print('Loading dataset done!')

    dataloader_a = torch.utils.data.DataLoader(trainset_a, batch_size=arg.batch_size, shuffle=arg.shuffle,
                                             num_workers=arg.workers, pin_memory=True)
    iterator_a = enumerate(dataloader_a)
    dataloader_b = torch.utils.data.DataLoader(trainset_b, batch_size=arg.batch_size, shuffle=arg.shuffle,
                                               num_workers=arg.workers, pin_memory=True)
    steps_per_epoch = len(dataloader_b)

    # evolving training
    print('Start training ...')
    for epoch in range(arg.resume_epoch, arg.max_epoch):
        global_step_base = epoch * steps_per_epoch
        forward_times_per_epoch, sum_loss_discrim, sum_loss_gen = 0, 0., 0.

        if epoch in arg.step_values:
            optimizer_decoder.param_groups[0]['lr'] *= arg.gamma

        for data_b in tqdm.tqdm(dataloader_b):
            try:
                data_a = next(iterator_a)[-1]
            except StopIteration:
                iterator_a = enumerate(dataloader_a)
                data_a = next(iterator_a)[-1]

            _, input_images_a, _, coords_a, _, _, _, _ = data_a
            _, input_images_b, _, coords_b, _, _, _, _ = data_b

            if input_images_a.size(0) != input_images_b.size(0):
                print('Resampling')
                continue

            forward_times_per_epoch += 1
            global_step = global_step_base + forward_times_per_epoch

            input_batch_size = input_images_a.shape[0]

            if arg.cuda:
                input_images_a = input_images_a.cuda(device=devices[0])
                input_images_b = input_images_b.cuda(device=devices[0])
                # coords_a = coords_a.cuda(device=devices[0])
                # coords_b = coords_b.cuda(device=devices[0])

            estimator.zero_grad()
            with torch.no_grad():
                heatmaps_a = estimator(input_images_a)[-1]
                heatmaps_b = estimator(input_images_b)[-1]
                # heatmaps_a[heatmaps_a < arg.boundary_cutoff_lambda * heatmaps_a.max()] = 0
                # heatmaps_b[heatmaps_b < arg.boundary_cutoff_lambda * heatmaps_b.max()] = 0

            with torch.no_grad():
                gen_ab = generator_a2b(heatmaps_a)
                gen_ba = generator_b2a(heatmaps_b)

            if arg.GAN:
                # discriminator optimize
                discrim_a.zero_grad()
                discrim_b.zero_grad()
                optimizer_discrim.zero_grad()

                loss_discrim_a_true = criterion_gan(discrim_a(edge(heatmaps_a)), discrim_true[:input_batch_size, ...])
                loss_discrim_a_false = criterion_gan(discrim_a(edge(gen_ba)), discrim_false[:input_batch_size, ...])
                loss_discrim_a = loss_discrim_a_true + loss_discrim_a_false
                log('loss_discrim_a_true', loss_discrim_a_true.item(), global_step)
                log('loss_discrim_a_false', loss_discrim_a_false.item(), global_step)

                loss_discrim_b_true = criterion_gan(discrim_b(edge(heatmaps_b)), discrim_true[:input_batch_size, ...])
                loss_discrim_b_false = criterion_gan(discrim_b(edge(gen_ab)), discrim_false[:input_batch_size, ...])
                loss_discrim_b = loss_discrim_b_true + loss_discrim_b_false
                log('loss_discrim_b_true', loss_discrim_b_true.item(), global_step)
                log('loss_discrim_b_false', loss_discrim_b_false.item(), global_step)

                loss_discrim = loss_discrim_a + loss_discrim_b
                log('loss_discrim_a', loss_discrim_a.item(), global_step)
                log('loss_discrim_b', loss_discrim_b.item(), global_step)
                log('loss_discrim', loss_discrim.item(), global_step)
                loss_discrim.backward()
                optimizer_discrim.step()

            # generator optimize
            pca.zero_grad()
            align.zero_grad()
            generator_a2b.zero_grad()
            generator_b2a.zero_grad()
            optimizer_generator.zero_grad()

            gen_ab = generator_a2b(heatmaps_a)
            gen_ba = generator_b2a(heatmaps_b)
            gen_aba = generator_b2a(gen_ab)
            gen_bab = generator_a2b(gen_ba)

            if arg.GAN:
                loss_gan_ab = criterion_gan(discrim_b(edge(gen_ab)), discrim_true[:input_batch_size, ...])
                loss_gan_ba  = criterion_gan(discrim_a(edge(gen_ba)), discrim_true[:input_batch_size, ...])
                loss_gan = loss_gan_ab + loss_gan_ba
                log('loss_gan_ab', loss_gan_ab.item(), global_step)
                log('loss_gan_ba', loss_gan_ba.item(), global_step)
                log('loss_gan', loss_gan.item(), global_step)

            loss_cycle_aba = criterion_pixel(gen_aba, heatmaps_a)
            loss_cycle_bab = criterion_pixel(gen_bab, heatmaps_b)
            loss_cycle = loss_cycle_aba + loss_cycle_bab
            log('loss_cycle_aba', loss_cycle_aba.item(), global_step)
            log('loss_cycle_bab', loss_cycle_bab.item(), global_step)
            log('loss_cycle', loss_cycle.item(), global_step)

            if criterion_gp is not None:
                loss_gp_aba = calc_heatmap_loss_gp(criterion_gp, gen_aba, heatmaps_a)
                loss_gp_bab = calc_heatmap_loss_gp(criterion_gp, gen_bab, heatmaps_b)
                loss_gp = loss_gp_aba + loss_gp_bab
                log('loss_gp', loss_gp.item(), global_step)

            pca_gen_ab = pca(align(gen_ab))
            pca_gen_ba = pca(align(gen_ba))
            pca_coords_a = pca(align(heatmaps_a)).detach()
            pca_coords_b = pca(align(heatmaps_b)).detach()

            loss_pca_ab = criterion_pca(pca_gen_ab[:, :arg.pca_used_components], pca_coords_a[:, :arg.pca_used_components])
            loss_pca_ba = criterion_pca(pca_gen_ba[:, :arg.pca_used_components], pca_coords_b[:, :arg.pca_used_components])
            loss_pca = loss_pca_ab + loss_pca_ba
            log('loss_pca_ab', loss_pca_ab.item(), global_step)
            log('loss_pca_ba', loss_pca_ba.item(), global_step)
            log('loss_pca', loss_pca.item(), global_step)

            loss_gen = arg.loss_cycle_lambda * loss_cycle + arg.loss_pca_lambda * loss_pca
            if arg.GAN:
                loss_gen = loss_gen + arg.loss_discrim_lambda * loss_gan

            if criterion_gp is not None:
                loss_gen = loss_gen + arg.loss_gp_lambda * loss_gp

            log('loss_gen', loss_gen.item(), global_step)
            loss_gen.backward()
            optimizer_generator.step()

            if arg.GAN:
                sum_loss_discrim += loss_discrim.item()

            sum_loss_gen += loss_gen.item()

            if arg.save_logs:
                heatmaps_a_to_save = get_heatmap_gray(heatmaps_a[0].unsqueeze(0), denorm=True).detach().cpu()
                heatmaps_b_to_save = get_heatmap_gray(heatmaps_b[0].unsqueeze(0), denorm=True).detach().cpu()
                heatmaps_a_edge_to_save = get_heatmap_gray(edge(heatmaps_a[0].unsqueeze(0)), denorm=True).detach().cpu()
                heatmaps_b_edge_to_save = get_heatmap_gray(edge(heatmaps_b[0].unsqueeze(0)), denorm=True).detach().cpu()
                gen_ab_to_save = get_heatmap_gray(gen_ab[0].unsqueeze(0), denorm=True).detach().cpu()
                gen_ba_to_save = get_heatmap_gray(gen_ba[0].unsqueeze(0), denorm=True).detach().cpu()
                gen_aba_to_save = get_heatmap_gray(gen_aba[0].unsqueeze(0), denorm=True).detach().cpu()
                gen_bab_to_save = get_heatmap_gray(gen_bab[0].unsqueeze(0), denorm=True).detach().cpu()
                heatmaps_to_save = make_grid(torch.stack([heatmaps_a_to_save,
                                       heatmaps_b_to_save,
                                       heatmaps_a_edge_to_save,
                                       heatmaps_b_edge_to_save,
                                       gen_ab_to_save,
                                       gen_ba_to_save,
                                       gen_aba_to_save,
                                       gen_bab_to_save]), pad_value=255)
                log_img('heatmaps', heatmaps_to_save, global_step)

        if arg.GAN:
            mean_sum_loss_discrim = sum_loss_discrim / forward_times_per_epoch

        mean_sum_loss_gen = sum_loss_gen / forward_times_per_epoch

        # if arg.GAN:
        #     scheduler_discrim.step(mean_sum_loss_discrim)
        #
        # scheduler_generator.step(mean_sum_loss_gen)

        if (epoch+1) % arg.save_interval == 0:
            torch.save(generator_a2b.state_dict(),
                       arg.save_folder + 'transformer_' + arg.dataset + '_' + arg.split_source + '2' + arg.split + '_' + str(epoch+1) + '.pth')
            torch.save(generator_b2a.state_dict(),
                       arg.save_folder + 'transformer_' + arg.dataset + '_' + arg.split + '2' + arg.split_source + '_' + str(epoch + 1) + '.pth')

            if arg.GAN:
                torch.save(discrim_a.state_dict(),
                           arg.save_folder + 'transformer_discrim_' + arg.dataset + '_' + arg.split_source + '_' + str(epoch + 1) + '.pth')
                torch.save(discrim_b.state_dict(),
                           arg.save_folder + 'transformer_discrim_' + arg.dataset + '_' + arg.split + '_' + str(epoch + 1) + '.pth')

        # if log_writer is not None:
        #     log_writer.add_scalar()

        print('\nepoch: {:0>4d} | loss_gen: {:.6f} '.format(
            epoch,
            mean_sum_loss_gen
        ))

    torch.save(generator_a2b.state_dict(),
               arg.save_folder + 'transformer_' + arg.dataset + '_' + arg.split_source + '2' + arg.split + '_' + str(
                   epoch + 1) + '.pth')
    torch.save(generator_b2a.state_dict(),
               arg.save_folder + 'transformer_' + arg.dataset + '_' + arg.split + '2' + arg.split_source + '_' + str(
                   epoch + 1) + '.pth')
    if arg.GAN:
        torch.save(discrim_a.state_dict(),
                   arg.save_folder + 'transformer_discrim_' + arg.dataset + '_' + arg.split_source + '_' + str(
                       epoch + 1) + '.pth')
        torch.save(discrim_b.state_dict(),
                   arg.save_folder + 'transformer_discrim_' + arg.dataset + '_' + arg.split + '_' + str(epoch + 1) + '.pth')
    print('Training done!')


if __name__ == '__main__':
    arg = parse_args()

    if not os.path.exists(arg.save_folder):
        os.mkdir(arg.save_folder)
    if not os.path.exists(arg.resume_folder):
        os.mkdir(arg.resume_folder)

    train(arg)