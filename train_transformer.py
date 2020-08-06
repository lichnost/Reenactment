import os
import torch
import torch.nn as nn
from utils.dataset import GeneralDataset, ShapePCADataset, ShapeFlameDataset
from utils import *
from utils.args import parse_args
import tqdm
from kornia.color import denormalize, normalize, rgb_to_grayscale
from kornia import image_to_tensor, scale, translate, resize
from models import GPLoss
from flame.FLAME import get_flame_layer, render_images, random_texture

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import json



def train_preliminary(arg):
    log_writer = None
    if arg.save_logs:
        log_path = './logs/transformer_' + arg.dataset + '_' + arg.split_source + '2' + arg.split
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

    pca_inverse = create_model_pca(arg, devices, eval=True, inverse=True)
    pca_inverse.eval()

    align = create_model_align(arg, devices, eval=True)
    align.eval()

    edge = create_model_edge(arg, devices, eval=True)
    edge.eval()

    generator_a2b = create_model_transformer_a2b(arg, devices, eval=False)
    # print_network(generator_a2b)
    generator_a2b.train()
    generator_b2a = create_model_transformer_b2a(arg, devices, eval=False)
    generator_b2a.train()

    if arg.GAN:
        discrim_a = create_model_transformer_discrim_a(arg, devices, eval=False)
        # print_network(discrim_a)
        discrim_a.train()
        discrim_b = create_model_transformer_discrim_b(arg, devices, eval=False)
        discrim_b.train()

    print('Creating networks done!')

    optimizer_generator, scheduler_generator = create_optimizer(arg, list(generator_a2b.parameters()) + list(generator_b2a.parameters()))

    if arg.GAN:
        optimizer_discrim, scheduler_discrim = create_optimizer(arg, list(discrim_a.parameters()) + list(discrim_b.parameters()), discrim=True)

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
        discrim_true_train = torch.ones((arg.batch_size, 1, 6, 6)).fill_(0.85)
        discrim_true = torch.ones((arg.batch_size, 1, 6, 6))
        if arg.cuda:
            criterion_gan = criterion_gan.cuda(device=devices[0])

            discrim_false = discrim_false.cuda(device=devices[0])
            discrim_true_train = discrim_true_train.cuda(device=devices[0])
            discrim_true = discrim_true.cuda(device=devices[0])

    # criterion_pixel = nn.L1Loss()
    # if arg.cuda:
    #     criterion_pixel = criterion_pixel.cuda(device=devices[0])

    criterion_gp = GPLoss()
    if arg.cuda:
        criterion_gp = criterion_gp.cuda(device=devices[0])

    criterion_pca = nn.L1Loss()
    if arg.cuda:
        criterion_pca = criterion_pca.cuda(device=devices[0])

    print('Loading dataset ...')

    trainset_b = ShapePCADataset(arg, dataset=arg.dataset, split=arg.split)
    trainset_a = ShapePCADataset(arg, dataset=arg.dataset_source, split=arg.split_source, trainset_sim=trainset_b)

    dataloader_a = torch.utils.data.DataLoader(trainset_a, batch_size=arg.batch_size, shuffle=arg.shuffle,
                                             num_workers=arg.workers, pin_memory=True)

    steps_per_epoch = len(dataloader_a)
    print('Loading dataset done!')

    # evolving training
    print('Start training ...')
    for epoch in range(arg.resume_epoch, arg.max_epoch):
        global_step_base = epoch * steps_per_epoch
        forward_times_per_epoch, sum_loss_discrim, sum_loss_gen = 0, 0., 0.

        if epoch in arg.step_values:
            optimizer_decoder.param_groups[0]['lr'] *= arg.gamma

        for data_a in tqdm.tqdm(dataloader_a):
            forward_times_per_epoch += 1
            global_step = global_step_base + forward_times_per_epoch

            shapes_a, heatmaps_a, _, _, _ = data_a
            _, heatmaps_b, _, _, _ = trainset_a.get_similars(shapes_a.detach().cpu().numpy())

            input_batch_size = shapes_a.shape[0]

            if arg.cuda:
                heatmaps_a = heatmaps_a.cuda(device=devices[0])
                heatmaps_b = heatmaps_b.cuda(device=devices[0])

            edge.zero_grad()

            if arg.GAN:
                # discriminator optimize
                discrim_a.train()
                discrim_b.train()
                generator_a2b.eval()
                generator_b2a.eval()

                discrim_a.zero_grad()
                discrim_b.zero_grad()
                optimizer_discrim.zero_grad()


                with torch.no_grad():
                    gen_ab = generator_a2b(heatmaps_a)
                    gen_ba = generator_b2a(heatmaps_b)

                loss_discrim_a_true = criterion_gan(discrim_a(edge(heatmaps_a)), discrim_true_train[:input_batch_size, ...])
                loss_discrim_a_false = criterion_gan(discrim_a(edge(gen_ba)), discrim_false[:input_batch_size, ...])
                loss_discrim_a = loss_discrim_a_true + loss_discrim_a_false
                log('loss_discrim_a_true', loss_discrim_a_true.item(), global_step)
                log('loss_discrim_a_false', loss_discrim_a_false.item(), global_step)

                loss_discrim_b_true = criterion_gan(discrim_b(edge(heatmaps_b)), discrim_true_train[:input_batch_size, ...])
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
            generator_a2b.train()
            generator_b2a.train()
            pca.zero_grad()
            align.zero_grad()
            generator_a2b.zero_grad()
            generator_b2a.zero_grad()
            optimizer_generator.zero_grad()

            gen_ab = generator_a2b(heatmaps_a)
            gen_ba = generator_b2a(heatmaps_b)
            gen_aba = generator_b2a(gen_ab)
            gen_bab = generator_a2b(gen_ba)

            edge_ab = edge(gen_ab)
            edge_ba = edge(gen_ba)

            if arg.GAN:
                discrim_a.eval()
                discrim_b.eval()

                loss_gan_ab = criterion_gan(discrim_b(edge_ab), discrim_true[:input_batch_size, ...])
                loss_gan_ba  = criterion_gan(discrim_a(edge_ba), discrim_true[:input_batch_size, ...])
                loss_gan = loss_gan_ab + loss_gan_ba
                log('loss_gan_ab', loss_gan_ab.item(), global_step)
                log('loss_gan_ba', loss_gan_ba.item(), global_step)
                log('loss_gan', loss_gan.item(), global_step)

            loss_gp_ab = calc_heatmap_loss_gp(criterion_gp, gen_ab, heatmaps_b)
            loss_gp_ba = calc_heatmap_loss_gp(criterion_gp, gen_ba, heatmaps_a)

            loss_gp_aba = calc_heatmap_loss_gp(criterion_gp, gen_aba, heatmaps_a)
            loss_gp_bab = calc_heatmap_loss_gp(criterion_gp, gen_bab, heatmaps_b)

            loss_gp = loss_gp_ab + loss_gp_ba + loss_gp_aba + loss_gp_bab

            log('loss_gp_ab', loss_gp_ab.item(), global_step)
            log('loss_gp_ba', loss_gp_ba.item(), global_step)
            log('loss_gp_aba', loss_gp_aba.item(), global_step)
            log('loss_gp_bab', loss_gp_bab.item(), global_step)
            log('loss_gp', loss_gp.item(), global_step)

            pca_gen_ab = pca(align(edge_ab))
            pca_gen_ba = pca(align(edge_ba))
            pca_coords_a = pca(align(edge(heatmaps_a))).detach()
            pca_coords_b = pca(align(edge(heatmaps_b))).detach()

            loss_pca_ab = criterion_pca(pca_gen_ab[:, :arg.pca_used_components], pca_coords_a[:, :arg.pca_used_components])
            loss_pca_ba = criterion_pca(pca_gen_ba[:, :arg.pca_used_components], pca_coords_b[:, :arg.pca_used_components])
            loss_pca = loss_pca_ab + loss_pca_ba
            log('loss_pca_ab', loss_pca_ab.item(), global_step)
            log('loss_pca_ba', loss_pca_ba.item(), global_step)
            log('loss_pca', loss_pca.item(), global_step)

            loss_gen =  arg.loss_gp_lambda * loss_gp + arg.loss_pca_lambda * loss_pca
            if arg.GAN:
                loss_gen = loss_gen + arg.loss_discrim_lambda * loss_gan

            log('loss_gen', loss_gen.item(), global_step)
            loss_gen.backward()
            optimizer_generator.step()

            if arg.GAN:
                sum_loss_discrim += loss_discrim.item()

            sum_loss_gen += loss_gen.item()

            if arg.save_logs and arg.save_img:
                heatmaps_a_to_save = get_heatmap_gray(heatmaps_a[0].unsqueeze(0), cutoff=True).detach().cpu()
                heatmaps_b_to_save = get_heatmap_gray(heatmaps_b[0].unsqueeze(0), cutoff=True).detach().cpu()
                heatmaps_a_edge_to_save = get_heatmap_gray(edge(heatmaps_a[0].unsqueeze(0)), cutoff=True).detach().cpu()
                heatmaps_b_edge_to_save = get_heatmap_gray(edge(heatmaps_b[0].unsqueeze(0)), cutoff=True).detach().cpu()
                gen_ab_to_save = get_heatmap_gray(gen_ab[0].unsqueeze(0)).detach().cpu()
                gen_ba_to_save = get_heatmap_gray(gen_ba[0].unsqueeze(0)).detach().cpu()
                gen_aba_to_save = get_heatmap_gray(gen_aba[0].unsqueeze(0)).detach().cpu()
                gen_bab_to_save = get_heatmap_gray(gen_bab[0].unsqueeze(0)).detach().cpu()

                heatmaps_input_to_save = make_grid(torch.stack([heatmaps_a_to_save,
                                       heatmaps_b_to_save,
                                       heatmaps_a_edge_to_save,
                                       heatmaps_b_edge_to_save]))

                heatmaps_generated_to_save = make_grid(torch.stack([gen_ab_to_save,
                                                          gen_ba_to_save,
                                                          gen_aba_to_save,
                                                          gen_bab_to_save]), normalize=True)
                log_img('images', make_grid(torch.stack([heatmaps_input_to_save,
                                                         heatmaps_generated_to_save]),
                                            normalize=False,
                                            nrow=1,
                                            padding=0), global_step)

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

        print('\nepoch: {:0>4d} | loss_gen: {:.6f}'.format(
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


def train_fine(arg):
    log_writer = None
    if arg.save_logs:
        log_path = './logs/transformer_' + arg.dataset + '_' + arg.split_source + '2' + arg.split
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

    pca_inverse = create_model_pca(arg, devices, eval=True, inverse=True)
    pca_inverse.eval()

    estimator = create_model_estimator(arg, devices, eval=True)
    estimator.eval()

    align = create_model_align(arg, devices, eval=True)
    align.eval()

    edge = create_model_edge(arg, devices, eval=True)
    edge.eval()

    generator_a2b = create_model_transformer_a2b(arg, devices, eval=False)
    # print_network(generator_a2b)
    generator_a2b.train()
    generator_b2a = create_model_transformer_b2a(arg, devices, eval=False)
    generator_b2a.train()

    if arg.GAN:
        discrim_a = create_model_transformer_discrim_a(arg, devices, eval=False)
        # print_network(discrim_a)
        discrim_a.train()
        discrim_b = create_model_transformer_discrim_b(arg, devices, eval=False)
        discrim_b.train()

    print('Creating networks done!')

    optimizer_generator, scheduler_generator = create_optimizer(arg, list(generator_a2b.parameters()) + list(generator_b2a.parameters()))

    if arg.GAN:
        optimizer_discrim, scheduler_discrim = create_optimizer(arg, list(discrim_a.parameters()) + list(discrim_b.parameters()), discrim=True)

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
        discrim_true_train = torch.ones((arg.batch_size, 1, 6, 6)).fill_(0.85)
        discrim_true = torch.ones((arg.batch_size, 1, 6, 6))
        if arg.cuda:
            criterion_gan = criterion_gan.cuda(device=devices[0])

            discrim_false = discrim_false.cuda(device=devices[0])
            discrim_true_train = discrim_true_train.cuda(device=devices[0])
            discrim_true = discrim_true.cuda(device=devices[0])

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

            estimator.zero_grad()
            edge.zero_grad()

            with torch.no_grad():
                heatmaps_a = estimator(input_images_a)[-1]
                heatmaps_b = estimator(input_images_b)[-1]

            if arg.GAN:
                # discriminator optimize
                discrim_a.train()
                discrim_b.train()
                generator_a2b.eval()
                generator_b2a.eval()

                discrim_a.zero_grad()
                discrim_b.zero_grad()
                optimizer_discrim.zero_grad()

                with torch.no_grad():
                    gen_ab = generator_a2b(heatmaps_a)
                    gen_ba = generator_b2a(heatmaps_b)

                loss_discrim_a_true = criterion_gan(discrim_a(edge(heatmaps_a)), discrim_true_train[:input_batch_size, ...])
                loss_discrim_a_false = criterion_gan(discrim_a(edge(gen_ba)), discrim_false[:input_batch_size, ...])
                loss_discrim_a = loss_discrim_a_true + loss_discrim_a_false
                log('loss_discrim_a_true', loss_discrim_a_true.item(), global_step)
                log('loss_discrim_a_false', loss_discrim_a_false.item(), global_step)

                loss_discrim_b_true = criterion_gan(discrim_b(edge(heatmaps_b)), discrim_true_train[:input_batch_size, ...])
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
            generator_a2b.train()
            generator_b2a.train()
            pca.zero_grad()
            align.zero_grad()
            generator_a2b.zero_grad()
            generator_b2a.zero_grad()
            optimizer_generator.zero_grad()

            gen_ab = generator_a2b(heatmaps_a)
            gen_ba = generator_b2a(heatmaps_b)
            gen_aba = generator_b2a(gen_ab)
            gen_bab = generator_a2b(gen_ba)

            edge_ab = edge(gen_ab)
            edge_ba = edge(gen_ba)

            if arg.GAN:
                discrim_a.eval()
                discrim_b.eval()

                loss_gan_ab = criterion_gan(discrim_b(edge_ab), discrim_true[:input_batch_size, ...])
                loss_gan_ba  = criterion_gan(discrim_a(edge_ba), discrim_true[:input_batch_size, ...])
                loss_gan = loss_gan_ab + loss_gan_ba
                log('loss_gan_ab', loss_gan_ab.item(), global_step)
                log('loss_gan_ba', loss_gan_ba.item(), global_step)
                log('loss_gan', loss_gan.item(), global_step)

            loss_gp_aba = calc_heatmap_loss_gp(criterion_gp, gen_aba, heatmaps_a)
            loss_gp_bab = calc_heatmap_loss_gp(criterion_gp, gen_bab, heatmaps_b)
            loss_gp = loss_gp_aba + loss_gp_bab
            log('loss_gp', loss_gp.item(), global_step)

            align_coords_ab = align(edge_ab)
            align_coords_ba = align(edge_ba)
            pca_gen_ab = pca(align_coords_ab)
            pca_gen_ba = pca(align_coords_ba)

            align_coords_a = align(edge(heatmaps_a)).detach()
            align_coords_b = align(edge(heatmaps_b)).detach()
            pca_coords_a = pca(align_coords_a)
            pca_coords_b = pca(align_coords_b)

            loss_pca_ab = criterion_pca(pca_gen_ab[:, :arg.pca_used_components], pca_coords_a[:, :arg.pca_used_components])
            loss_pca_ba = criterion_pca(pca_gen_ba[:, :arg.pca_used_components], pca_coords_b[:, :arg.pca_used_components])
            loss_pca = loss_pca_ab + loss_pca_ba
            log('loss_pca_ab', loss_pca_ab.item(), global_step)
            log('loss_pca_ba', loss_pca_ba.item(), global_step)
            log('loss_pca', loss_pca.item(), global_step)

            loss_gen =  arg.loss_gp_lambda * loss_gp + arg.loss_pca_lambda * loss_pca
            if arg.GAN:
                loss_gen = loss_gen + arg.loss_discrim_lambda * loss_gan

            log('loss_gen', loss_gen.item(), global_step)
            loss_gen.backward()
            optimizer_generator.step()

            if arg.GAN:
                sum_loss_discrim += loss_discrim.item()

            sum_loss_gen += loss_gen.item()

            if arg.save_logs and arg.save_img:
                heatmaps_a_to_save = get_heatmap_gray(heatmaps_a[0].unsqueeze(0), cutoff=True).detach().cpu()
                heatmaps_b_to_save = get_heatmap_gray(heatmaps_b[0].unsqueeze(0), cutoff=True).detach().cpu()
                heatmaps_a_edge_to_save = get_heatmap_gray(edge(heatmaps_a[0].unsqueeze(0)), cutoff=True).detach().cpu()
                heatmaps_b_edge_to_save = get_heatmap_gray(edge(heatmaps_b[0].unsqueeze(0)), cutoff=True).detach().cpu()
                gen_ab_to_save = get_heatmap_gray(gen_ab[0].unsqueeze(0)).detach().cpu()
                gen_ba_to_save = get_heatmap_gray(gen_ba[0].unsqueeze(0)).detach().cpu()
                gen_aba_to_save = get_heatmap_gray(gen_aba[0].unsqueeze(0)).detach().cpu()
                gen_bab_to_save = get_heatmap_gray(gen_bab[0].unsqueeze(0)).detach().cpu()

                heatmaps_input_to_save = make_grid(torch.stack([
                    heatmaps_a_to_save,
                    heatmaps_b_to_save,
                    heatmaps_a_edge_to_save,
                    heatmaps_b_edge_to_save
                ]))

                heatmaps_generated_to_save = make_grid(torch.stack([
                    gen_ab_to_save,
                    gen_ba_to_save,
                    gen_aba_to_save,
                    gen_bab_to_save
                ]), normalize=True)

                img_size = heatmaps_a.shape[2]
                align_coords_to_save = make_grid(torch.stack([
                    image_to_tensor(draw_coords(arg.dataset, img_size, align_coords_ab[0].detach().cpu().numpy())),
                    image_to_tensor(draw_coords(arg.dataset, img_size, align_coords_ba[0].detach().cpu().numpy())),
                    image_to_tensor(draw_coords(arg.dataset, img_size, align_coords_a[0].detach().cpu().numpy())),
                    image_to_tensor(draw_coords(arg.dataset, img_size, align_coords_b[0].detach().cpu().numpy()))
                ]), normalize=True)

                inv_coords_to_save = make_grid(torch.stack([
                    image_to_tensor(draw_coords(arg.dataset, img_size, pca_inverse(pca_gen_ab[0].unsqueeze(0))[0].detach().cpu().numpy())),
                    image_to_tensor(draw_coords(arg.dataset, img_size, pca_inverse(pca_gen_ba[0].unsqueeze(0))[0].detach().cpu().numpy())),
                    image_to_tensor(draw_coords(arg.dataset, img_size, pca_inverse(pca_coords_a[0].unsqueeze(0))[0].detach().cpu().numpy())),
                    image_to_tensor(draw_coords(arg.dataset, img_size, pca_inverse(pca_coords_b[0].unsqueeze(0))[0].detach().cpu().numpy()))
                ]), normalize=True)

                log_img('images', make_grid(torch.stack([heatmaps_input_to_save,
                                                         heatmaps_generated_to_save,
                                                         align_coords_to_save,
                                                         inv_coords_to_save]),
                                            normalize=False,
                                            nrow=1,
                                            padding=0), global_step)

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


def draw_coords(dataset, size, coords):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(0, kp_num[dataset] - 1):
        draw_circle(img, (int(coords[2 * i]), int(coords[2 * i + 1])))  # red
    return img


def train_flame(arg):
    log_writer = None
    if arg.save_logs:
        log_path = './logs/transformer_' + arg.dataset + '_' + arg.split_source + '2' + arg.split
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
          '# Use GPU:            ' + str(arg.cuda) + '\n' +
          '# Start lr:           ' + str(arg.lr) + '\n' +
          '# Max epoch:          ' + str(arg.max_epoch) + '\n' +
          '# Resumed model:      ' + str(arg.resume_epoch > 0))
    if arg.resume_epoch > 0:
        print('# Resumed epoch:      ' + str(arg.resume_epoch))

    # log_text('arguments', json.dumps(vars(arg), indent=2), 0)

    print('Creating networks ...')

    estimator = create_model_estimator(arg, devices, eval=True)
    estimator.eval()

    edge = create_model_edge(arg, devices, eval=True)
    edge.eval()

    generator_a2b = create_model_transformer_a2b(arg, devices, eval=False)
    # print_network(generator_a2b)
    generator_a2b.train()

    optimizer_generator_ab, scheduler_generator_ab = create_optimizer(arg, generator_a2b.parameters())

    print('Creating networks done!')

    print('Creating FLAME model layer ...')

    flame = get_flame_layer(arg.flame_model_path,
                            arg.flame_static_landmark_embedding_path,
                            arg.flame_dynamic_landmark_embedding_path,
                            arg.batch_size,
                            arg.flame_shape_params,
                            arg.flame_expression_params,
                            arg.flame_pose_params,
                            arg.flame_use_3D_translation)
    if arg.cuda:
        flame = flame.to(devices[0])

    faces = torch.from_numpy(np.float32(flame.faces))
    faces = torch.cat(arg.batch_size * [faces.unsqueeze(0)]).to(torch.int64)
    if arg.cuda:
        faces = faces.to(devices[0])

    texture_model = np.load(arg.flame_texture_path)
    texture, faces_uvs, verts_uvs = random_texture(texture_model, arg.batch_size)
    if arg.cuda:
        texture = texture.to(devices[0])
        faces_uvs = faces_uvs.to(devices[0])
        verts_uvs = verts_uvs.to(devices[0])

    print('Creating FLAME model layer done!')

    criterion_gp = GPLoss()
    if arg.cuda:
        criterion_gp = criterion_gp.cuda(device=devices[0])

    if arg.loss_type == 'L1':
        criterion = nn.L1Loss()
    elif arg.loss_type == 'smoothL1':
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()
    if arg.cuda:
        criterion = criterion.cuda(device=devices[0])

    criterion_edge = nn.SmoothL1Loss()
    if arg.cuda:
        criterion_edge = criterion_edge.cuda(device=devices[0])


    print('Loading dataset ...')

    trainset_a = ShapeFlameDataset(arg, dataset=arg.dataset_source, split=arg.split_source)
    trainset_b = ShapeFlameDataset(arg, dataset=arg.dataset, split=arg.split)

    dataloader_a = torch.utils.data.DataLoader(trainset_a, batch_size=arg.batch_size, shuffle=arg.shuffle,
                                               num_workers=arg.workers, pin_memory=False, drop_last=True,
                                               worker_init_fn=lambda _: np.random.seed())
    if arg.flame_shape_params_path_b is not None:
        params = np.load(path, allow_pickle=True, encoding='latin1').item()
        shape_params_b = torch.from_numpy(np.float32(params['shape']))
    else:
        shape_params_b = torch.from_numpy(trainset_b.mean_shape_params)
    shape_params_b = torch.cat(arg.batch_size * [shape_params_b.unsqueeze(0)])
    if arg.cuda:
        shape_params_b = shape_params_b.to(device=devices[0])

    shape_params_a = None
    if arg.flame_shape_params_path_a is not None:
        params = np.load(path, allow_pickle=True, encoding='latin1').item()
        shape_params_a = torch.from_numpy(np.float32(params['shape']))
        shape_params_a = torch.cat(arg.batch_size * [shape_params_a.unsqueeze(0)])
        if arg.cuda:
            shape_params_a = shape_params_a.to(device=devices[0])

    mean = torch.FloatTensor(means_color[arg.dataset][arg.split])
    std = torch.FloatTensor(stds_color[arg.dataset][arg.split])
    norm_min = (0 - mean) / std
    norm_max = (255 - mean) / std
    norm_range = norm_max - norm_min
    norm_range = torch.where(norm_range < 1e-6, torch.ones_like(norm_range), norm_range)

    mean_gray = means_gray[arg.dataset][arg.split]
    std_gray = stds_gray[arg.dataset][arg.split]

    if arg.cuda:
        mean = mean.cuda(device=devices[0])
        std = std.cuda(device=devices[0])
        norm_min = norm_min.cuda(device=devices[0])
        norm_max = norm_max.cuda(device=devices[0])

    print('Loading dataset done!')

    steps_per_epoch = len(dataloader_a)

    # evolving training
    print('Start training ...')
    for epoch in range(arg.resume_epoch, arg.max_epoch):
        global_step_base = epoch * steps_per_epoch
        forward_times_per_epoch, sum_loss = 0, 0.0

        for data in tqdm.tqdm(dataloader_a):
            forward_times_per_epoch += 1
            global_step = global_step_base + forward_times_per_epoch

            path, shape, pose, neck_pose, expression, transl, scl = data
            if shape_params_a is not None:
                shape = shape_params_a

            transl = transl * arg.crop_size * 2

            if arg.cuda:
                shape = shape.to(device=devices[0])
                # pose = torch.zeros_like(pose)
                pose = pose.to(device=devices[0])
                neck_pose = neck_pose.to(device=devices[0])
                expression = expression.to(device=devices[0])
                transl = transl.to(device=devices[0])
                scl = scl.to(device=devices[0])

            with torch.no_grad():
                vertices_a, _ = flame.forward(shape, expression, pose, neck_pose)
                images_a = render_images(vertices_a, faces, texture, faces_uvs, verts_uvs, arg.crop_size, device=devices[0])[..., :3]

                vertices_b, _ = flame.forward(shape_params_b, expression, pose, neck_pose)
                images_b = render_images(vertices_b, faces, texture, faces_uvs, verts_uvs, arg.crop_size, device=devices[0])[..., :3]

                images_a = rgb_to_grayscale(images_a.permute(0, 3, 1, 2))
                images_a = scale(images_a, scl)
                images_a = translate(images_a, transl)
                images_a = resize(images_a, arg.crop_size)
                # images_a = rescale_0_1(images_a)

                images_b = rgb_to_grayscale(images_b.permute(0, 3, 1, 2))
                images_b = scale(images_b, scl)
                images_b = translate(images_b, transl)
                images_b = resize(images_b, arg.crop_size)
                # images_b = rescale_0_1(images_b)

                # mean_a = torch.mean(images_a)
                # std_a = torch.std(images_a)
                # images_a = normalize(images_a, mean_a, std_a)
                #
                # mean_b = torch.mean(images_b)
                # std_b = torch.std(images_b)
                # images_b = normalize(images_b, mean_b, std_b)

                heatmaps_a = estimator(images_a)[-1]
                heatmaps_a = rescale_0_1(heatmaps_a, torch.min(heatmaps_a), torch.max(heatmaps_a))
                heatmaps_b = estimator(images_b)[-1]
                heatmaps_b = rescale_0_1(heatmaps_b, torch.min(heatmaps_b), torch.max(heatmaps_b))

                edges_b = edge(heatmaps_b).detach()


            heatmaps = generator_a2b(heatmaps_a)
            heatmaps = rescale_0_1(heatmaps, torch.min(heatmaps), torch.max(heatmaps))
            edges = edge(heatmaps)

            optimizer_generator_ab.zero_grad()

            loss_gp = calc_heatmap_loss_gp(criterion_gp, heatmaps, heatmaps_b)
            log('loss_gp', loss_gp.item(), global_step)

            loss_main = criterion(heatmaps, heatmaps_b)
            log('loss_main', loss_main.item(), global_step)

            loss_edge = criterion_edge(edges, edges_b)
            log('loss_edge', loss_edge.item(), global_step)

            loss = arg.loss_gp_lambda * loss_gp + loss_main + arg.loss_edge_lambda * loss_edge
            log('loss', loss.item(), global_step)

            loss.backward()
            optimizer_generator_ab.step()

            sum_loss += loss.item()

            mean_sum_loss = sum_loss / forward_times_per_epoch


            if arg.save_logs and arg.save_img:
                images_a_to_save = resize(images_a[0].unsqueeze(0), heatmaps.shape[-1]).squeeze(1).detach().cpu()
                images_b_to_save = resize(images_b[0].unsqueeze(0), heatmaps.shape[-1]).squeeze(1).detach().cpu()
                edges_a_to_save = get_heatmap_gray(heatmaps_a[0], cutoff=True).unsqueeze(0).detach().cpu()
                edges_b_to_save = get_heatmap_gray(heatmaps_b[0], cutoff=True).unsqueeze(0).detach().cpu()
                edge_to_save = get_heatmap_gray(heatmaps[0], cutoff=True).unsqueeze(0).detach().cpu()

                to_save = make_grid(torch.stack([
                    images_a_to_save,
                    images_b_to_save,
                    edges_a_to_save,
                    edges_b_to_save,
                    edge_to_save
                ]))

                log_img('images', to_save, global_step)

            # show_img(images_a[0].cpu().squeeze(0).numpy(), 'a', wait=1, keep=True)
            # show_img(images_b[0].cpu().squeeze(0).numpy(), 'b', wait=1, keep=True)
            # heatmap_show = get_heatmap_gray(edges_a[0].unsqueeze(0)).detach().cpu().numpy()
            # heatmap_show = (
            #         255 - np.uint8(255 * (heatmap_show - np.min(heatmap_show)) / np.ptp(heatmap_show)))
            # heatmap_show = np.moveaxis(heatmap_show, 0, -1)
            # heatmap_show = cv2.resize(heatmap_show, (256, 256))
            #
            # show_img(heatmap_show, 'heatmap_a', wait=1, keep=True)
            #
            # heatmap_show = get_heatmap_gray(edges_b[0].unsqueeze(0)).detach().cpu().numpy()
            # heatmap_show = (
            #         255 - np.uint8(255 * (heatmap_show - np.min(heatmap_show)) / np.ptp(heatmap_show)))
            # heatmap_show = np.moveaxis(heatmap_show, 0, -1)
            # heatmap_show = cv2.resize(heatmap_show, (256, 256))
            #
            # show_img(heatmap_show, 'heatmap_b', wait=1, keep=True)
            #
            # heatmap_show = get_heatmap_gray(edges[0].unsqueeze(0)).detach().cpu().numpy()
            # heatmap_show = (
            #         255 - np.uint8(255 * (heatmap_show - np.min(heatmap_show)) / np.ptp(heatmap_show)))
            # heatmap_show = np.moveaxis(heatmap_show, 0, -1)
            # heatmap_show = cv2.resize(heatmap_show, (256, 256))
            #
            # show_img(heatmap_show, 'heatmap_gen')

            # scheduler_generator.step(mean_sum_loss_gen)

            if (epoch + 1) % arg.save_interval == 0:
                torch.save(generator_a2b.state_dict(),
                           arg.save_folder + 'transformer_' + arg.dataset + '_' + arg.split_source + '2' + arg.split + '_' + str(
                               epoch + 1) + '.pth')

            # if log_writer is not None:
            #     log_writer.add_scalar()

        print('\nepoch: {:0>4d} | loss: {:.6f} '.format(
            epoch,
            mean_sum_loss
        ))

    torch.save(generator_a2b.state_dict(),
               arg.save_folder + 'transformer_' + arg.dataset + '_' + arg.split_source + '2' + arg.split + '_' + str(
                   epoch + 1) + '.pth')
    print('Training done!')


if __name__ == '__main__':
    arg = parse_args()

    if not os.path.exists(arg.save_folder):
        os.mkdir(arg.save_folder)
    if not os.path.exists(arg.resume_folder):
        os.mkdir(arg.resume_folder)

    train_flame(arg)
