from utils import *
from utils.args import parse_args
from flame.FLAME import get_flame_layer, render_images, random_texture, get_texture_model
import cv2, torch
from kornia import image_to_tensor, tensor_to_image, rgb_to_bgr, denormalize, normalize, bgr_to_rgb, resize
from utils.dataset import DecoderDataset
from models import GPLoss, CPLoss
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def fit_flame(arg):
    log_writer = None
    if arg.save_logs:
        log_path = './logs/flame_' + arg.dataset + '_' + arg.split
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


    devices = get_devices_list(arg)

    print('Creating models...')
    flame = create_model_flame(arg, devices)

    segmentation = create_model_segment(arg, devices, eval=True)
    segmentation.eval()
    if arg.cuda:
        segmentation = segmentation.cuda(device=devices[0])

    print('Creating models done!')

    print('Loading dataset ...')
    trainset = DecoderDataset(arg, dataset=arg.dataset, split=arg.split)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, drop_last=True,
                                             num_workers=arg.workers, pin_memory=True)
    print('Loading dataset done!')

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

    criterion_simple = nn.SmoothL1Loss()
    if arg.cuda:
        criterion_simple = criterion_simple.cuda(device=devices[0])

    criterion_landmarks = nn.MSELoss()
    if arg.cuda:
        criterion_landmarks = criterion_landmarks.cuda(device=devices[0])

    parameters = flame.parameters()
    optimizer = torch.optim.LBFGS(parameters, tolerance_change=1e-10, max_iter=arg.max_epoch)

    mean = torch.FloatTensor(means_color[arg.dataset][arg.split])
    std = torch.FloatTensor(stds_color[arg.dataset][arg.split])
    norm_min = (0 - mean) / std
    norm_max = (255 - mean) / std
    norm_range = norm_max - norm_min
    norm_range = torch.where(norm_range < 1e-6, torch.ones_like(norm_range), norm_range)

    if arg.cuda:
        mean = mean.cuda(device=devices[0])
        std = std.cuda(device=devices[0])
        norm_min = norm_min.cuda(device=devices[0])
        # norm_max = norm_max.cuda(device=devices[0])
        norm_range = norm_range.cuda(device=devices[0])

    iterator = iter(dataloader)
    input_images, _, original_landmarks = next(iterator)
    # input_landmarks = coords_seq_to_xy(arg.dataset, input_landmarks)
    # show_img(np.uint8(tensor_to_image(rgb_to_bgr(denormalize(input_images_norm[0], mean, std)))), 'target')

    if arg.cuda:
        input_images = input_images.to(device=devices[0])
        input_landmarks = original_landmarks.to(device=devices[0])
        # input_images_norm = input_images_norm.to(device=devices[0])
        # gt_coords_xy = gt_coords_xy.to(device=devices[0])

    masks = F.softmax(segmentation(input_images), dim=1)[:, 0, ...].unsqueeze(1).expand(-1, 3, -1, -1) * 255
    input_images = denormalize(input_images, mean, std)
    input_images = torch.clamp(input_images + masks, 0, 255)

    # scale = 4
    #
    # original = input_images[0].detach().unsqueeze(0)
    # # original = resize(original, original.size(-1) * 4)
    # image = np.uint8(tensor_to_image(rgb_to_bgr(original)))
    #
    # image = cv2.resize(image, (image.shape[0] * scale, image.shape[1] * scale))

    def get_original_landmarks(landmarks):
        result = torch.zeros([landmarks.size(0), 7, 2], device=landmarks.device)
        result[:, 0] = landmarks[:, nose_tip_x[arg.dataset]:nose_tip_y[arg.dataset]+1]
        result[:, 1] = landmarks[:, left_eye_left_corner_index_x[arg.dataset]:left_eye_left_corner_index_y[arg.dataset] + 1]
        result[:, 2] = landmarks[:, left_eye_right_corner_index_x[arg.dataset]:left_eye_right_corner_index_y[arg.dataset] + 1]
        result[:, 3] = landmarks[:, right_eye_left_corner_index_x[arg.dataset]:right_eye_left_corner_index_y[arg.dataset] + 1]
        result[:, 4] = landmarks[:, right_eye_right_corner_index_x[arg.dataset]:right_eye_right_corner_index_y[arg.dataset] + 1]
        result[:, 5] = landmarks[:, left_mouth_x[arg.dataset][0]:left_mouth_y[arg.dataset][0] + 1]
        result[:, 6] = landmarks[:, right_mouth_x[arg.dataset][0]:right_mouth_y[arg.dataset][0] + 1]
        return result

    original_landmarks = get_original_landmarks(input_landmarks)
    # original_landmarks = coords_seq_to_xy(arg.dataset, original_landmarks)
    # for idx in range(original_landmarks[0].size(0)):
    #     draw_circle(image, (int(original_landmarks[0, idx, 0] * scale), int(original_landmarks[0, idx, 1] * scale)))
    #     draw_text(image, str(idx), (int(original_landmarks[0, idx, 0]*scale), int(original_landmarks[0, idx, 1]*scale)))
    # show_img(image, 'original')

    # _, target_landmarks, rendered = flame()
    # rendered = derescale_0_1(rendered, 0, 255)
    #
    # target = rendered[0].detach().unsqueeze(0)
    # # target = resize(target, target.size(-1) * 2)
    # image = np.uint8(tensor_to_image(rgb_to_bgr(target))).copy()
    #
    # image = cv2.resize(image, (image.shape[0] * scale, image.shape[1] * scale))

    def get_target_landmarks(landmarks):
        result = torch.zeros([landmarks.size(0), 7, 2], device=landmarks.device)
        result[:, 0] = landmarks[:, 13]
        result[:, 1] = landmarks[:, 19]
        result[:, 2] = landmarks[:, 22]
        result[:, 3] = landmarks[:, 25]
        result[:, 4] = landmarks[:, 28]
        result[:, 5] = landmarks[:, 31]
        result[:, 6] = landmarks[:, 37]
        return result

    # target_landmarks = get_target_landmarks(target_landmarks)
    # for idx in range(target_landmarks[0].size(0)):
    #     draw_circle(image, (int(target_landmarks[0, idx, 0] * scale), int(target_landmarks[0, idx, 1] * scale)))
    #     draw_text(image, str(idx), (int(target_landmarks[0, idx, 0] * scale), int(target_landmarks[0, idx, 1] * scale)))
    # show_img(image, 'target')

    global_step = 0

    def closure():
        nonlocal global_step
        global_step += 1

        if torch.is_grad_enabled():
            optimizer.zero_grad()

        _, target_landmarks, rendered = flame()
        rendered = derescale_0_1(rendered, 0, 255)

        loss_landmarks = 50 * criterion_landmarks(get_target_landmarks(target_landmarks), original_landmarks)
        loss = loss_landmarks
        log('loss_landmarks', loss_landmarks.item(), global_step)

        loss_simple = criterion_simple(rendered, input_images)
        loss = loss + loss_simple
        log('loss_simple', loss_simple.item(), global_step)

        if criterion_gp is not None:
            loss_gp = arg.loss_gp_lambda * criterion_gp(rendered, input_images) + 100
            loss = loss + loss_gp
            log('loss_gp', loss_gp.item(), global_step)

        if criterion_cp is not None:
            loss_cp = arg.loss_cp_lambda * criterion_cp(rendered, input_images) + 100
            loss = loss + loss_cp
            log('loss_cp', loss_cp.item(), global_step)

        log('loss', loss.item(), global_step)

        if loss.requires_grad:
            loss.backward(retain_graph=True)

        log_img('original', input_images[0].detach().to(dtype=torch.uint8), global_step)
        log_img('image', rendered[0].detach().to(dtype=torch.uint8), global_step)

        return loss

    optimizer.step(closure)

    _, _, final_images = flame()



if __name__ == '__main__':
    arg = parse_args()

    if not os.path.exists(arg.save_folder):
        os.mkdir(arg.save_folder)
    if not os.path.exists(arg.resume_folder):
        os.mkdir(arg.resume_folder)

    fit_flame(arg)
