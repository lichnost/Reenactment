from .dataset_info import *
from .pdb import pdb
from .visual import *

import cv2
import time
import random
import numpy as np
from scipy.interpolate import splprep, splev
import torch

def get_annos_path(dataset_route, dataset, split):
    return dataset_route + dataset + '_' + split.replace('/', '-') + '_annos.txt'


def get_annotations_list(dataset_route, dataset, split, crop_size, ispdb=False):
    annotations = []
    annotation_file = open(get_annos_path(dataset_route[dataset], dataset, split))

    for line in range(dataset_size[dataset][split]):
        annotations.append(annotation_file.readline().rstrip().split())
    annotation_file.close()

    if ispdb:
        annos = []
        allshapes = np.zeros((2 * kp_num[dataset], len(annotations)))
        for line_index, line in enumerate(annotations):
            coord_x = np.array(list(map(float, line[:2*kp_num[dataset]:2])))
            coord_y = np.array(list(map(float, line[1:2*kp_num[dataset]:2])))
            position_before = np.float32([[int(line[-7]), int(line[-6])],
                                          [int(line[-7]), int(line[-4])],
                                          [int(line[-5]), int(line[-4])]])
            position_after = np.float32([[0, 0],
                                         [0, crop_size - 1],
                                         [crop_size - 1, crop_size - 1]])
            crop_matrix = cv2.getAffineTransform(position_before, position_after)
            coord_x_after_crop = crop_matrix[0][0] * coord_x + crop_matrix[0][1] * coord_y + crop_matrix[0][2]
            coord_y_after_crop = crop_matrix[1][0] * coord_x + crop_matrix[1][1] * coord_y + crop_matrix[1][2]
            allshapes[0:kp_num[dataset], line_index] = list(coord_x_after_crop)
            allshapes[kp_num[dataset]:2*kp_num[dataset], line_index] = list(coord_y_after_crop)
        newidx = pdb(dataset, allshapes, dataset_pdb_numbins[dataset])
        for id_index in newidx:
            annos.append(annotations[int(id_index)])
        return annos

    return annotations


def convert_img_to_gray(img):
    if img.shape[2] == 1:
        return img
    elif img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        return gray
    elif img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray
    else:
        raise Exception("img shape wrong!\n")


def get_random_transform_param(type, bbox, trans_ratio, rotate_limit, scale_ratio_up, scale_ratio_down, scale_horizontal, scale_vertical, flip=False, gaussian=True):
    translation = 0
    trans_dir = 0
    rotation = 0
    scaling = 1.0
    scaling_horizontal = 0.0
    scaling_vertical = 0.0
    flip_num = 0
    gaussian_blur = 0
    if type in ['train']:
        random.seed(time.time())
        translate_param = int(trans_ratio * abs(bbox[2] - bbox[0]))
        translation = random.randint(-translate_param, translate_param)
        trans_dir = random.randint(0, 3)  # LU:0 RU:1 LL:2 RL:3
        rotation = random.uniform(-rotate_limit, rotate_limit)
        scaling = random.uniform(1-scale_ratio_down, 1+scale_ratio_up)
        if random.randint(0, 1) == 0:
            scaling_horizontal = random.uniform(scale_horizontal, scale_horizontal)
        else:
            scaling_vertical = random.uniform(scale_vertical, scale_vertical)
        flip_num = random.randint(0, 1) if flip else 0
        gaussian_blur = random.randint(0, 1) if gaussian else 0
    return translation, trans_dir, rotation, scaling, scaling_horizontal, scaling_vertical, flip_num, gaussian_blur


def further_transform(pic, bbox, flip, gaussian_blur):
    if flip == 1:
        pic = cv2.flip(pic, 1)
    if abs(bbox[2] - bbox[0]) < 120 or gaussian_blur == 0:
        return pic
    else:
        return cv2.GaussianBlur(pic, (5, 5), 1)


def get_affine_matrix(width_height, rotation, scaling):
    center = (width_height[0] / 2.0, width_height[1] / 2.0)
    return cv2.getRotationMatrix2D(center, rotation, scaling)


def pic_normalize_gray_single(pic, mean=None, std=None):
    pic = np.float32(pic)
    if mean is None:
        mean = pic.mean()
    if std is None:
        std = pic.std()
    std = np.where(std < 1e-6, 1, std)
    pic = (pic - mean) / std
    return np.float32(pic)


def pic_normalize_gray(pic, mean=None, std=None):
    # if len(pic.shape) == 4:
    #     return np.apply_along_axis(pic_normalize_gray_single, 0)
    # else:
    return pic_normalize_gray_single(pic, mean, std)


def pic_normalize_color_single(pic, mean=None, std=None):
    pic = np.float32(pic)
    if mean is None:
        mean = pic.mean(axis=(0, 1, 2), keepdims=True)
    if std is None:
        std = pic.std(axis=(0, 1, 2), keepdims=True)
    std = np.where(std < 1e-6, 1, std)
    pic = (pic - mean) / std
    return pic


def pic_normalize_color(pic, mean=None, std=None):
    # if len(pic.shape) == 4:
    #     np.apply_along_axis(pic_normalize_color_single, 0)
    # else:
    return pic_normalize_color_single(pic, mean, std)


def get_cropped_coords(dataset, crop_matrix, coord_x, coord_y, crop_size, flip=0):
    coord_x, coord_y = np.array(coord_x), np.array(coord_y)
    temp_x = crop_matrix[0][0] * coord_x + crop_matrix[0][1] * coord_y + crop_matrix[0][2] if flip == 0 else \
        float(crop_size) - (crop_matrix[0][0] * coord_x + crop_matrix[0][1] * coord_y + crop_matrix[0][2]) - 1
    temp_y = crop_matrix[1][0] * coord_x + crop_matrix[1][1] * coord_y + crop_matrix[1][2]
    if flip:
        temp_x = temp_x[np.array(flip_relation[dataset])[:, 1]]
        temp_y = temp_y[np.array(flip_relation[dataset])[:, 1]]
    return temp_x, temp_y


def get_gt_coords(dataset, affine_matrix, coord_x, coord_y):
    out = np.zeros(2*kp_num[dataset])
    out[:2*kp_num[dataset]:2] = affine_matrix[0][0] * coord_x + affine_matrix[0][1] * coord_y + affine_matrix[0][2]
    out[1:2*kp_num[dataset]:2] = affine_matrix[1][0] * coord_x + affine_matrix[1][1] * coord_y + affine_matrix[1][2]
    return np.array(np.float32(out))


def get_gt_heatmap(dataset, gt_coords, crop_size, sigma):
    coord_x, coord_y, gt_heatmap = [], [], []
    for index in range(boundary_num):
        gt_heatmap.append(np.ones((heatmap_size, heatmap_size)))
        gt_heatmap[index].tolist()
    boundary_x = {'chin': [], 'leb': [], 'reb':  [], 'bon':  [], 'breath': [], 'lue':  [], 'lle': [],
                  'rue':  [], 'rle': [], 'usul': [], 'lsul': [], 'usll':   [], 'lsll': []}
    boundary_y = {'chin': [], 'leb': [], 'reb':  [], 'bon':  [], 'breath': [], 'lue':  [], 'lle': [],
                  'rue':  [], 'rle': [], 'usul': [], 'lsul': [], 'usll':   [], 'lsll': []}
    points = {'chin': [], 'leb': [], 'reb':  [], 'bon':  [], 'breath': [], 'lue':  [], 'lle': [],
              'rue':  [], 'rle': [], 'usul': [], 'lsul': [], 'usll':   [], 'lsll': []}
    resize_matrix = cv2.getAffineTransform(np.float32([[0, 0], [0, crop_size-1],
                                                       [crop_size-1, crop_size-1]]),
                                           np.float32([[0, 0], [0, heatmap_size-1],
                                                       [heatmap_size-1, heatmap_size-1]]))
    for kp_index in range(kp_num[dataset]):
        coord_x.append(
            resize_matrix[0][0] * gt_coords[2 * kp_index] +
            resize_matrix[0][1] * gt_coords[2 * kp_index + 1] +
            resize_matrix[0][2] + random.uniform(-0.2, 0.2)
        )
        coord_y.append(
            resize_matrix[1][0] * gt_coords[2 * kp_index] +
            resize_matrix[1][1] * gt_coords[2 * kp_index + 1] +
            resize_matrix[1][2] + random.uniform(-0.2, 0.2)
        )
    for boundary_index in range(boundary_num):
        for kp_index in range(
                point_range[dataset][boundary_index][0],
                point_range[dataset][boundary_index][1]
        ):
            boundary_x[boundary_keys[boundary_index]].append(coord_x[kp_index])
            boundary_y[boundary_keys[boundary_index]].append(coord_y[kp_index])
        if boundary_keys[boundary_index] in boundary_special.keys() and\
                dataset in boundary_special[boundary_keys[boundary_index]]:
            boundary_x[boundary_keys[boundary_index]].append(
                coord_x[duplicate_point[dataset][boundary_keys[boundary_index]]])
            boundary_y[boundary_keys[boundary_index]].append(
                coord_y[duplicate_point[dataset][boundary_keys[boundary_index]]])
    for k_index, k in enumerate(boundary_keys):
        if point_num_per_boundary[dataset][k_index] >= 2.:
            if len(boundary_x[k]) == len(set(boundary_x[k])) or len(boundary_y[k]) == len(set(boundary_y[k])):
                points[k].append(boundary_x[k])
                points[k].append(boundary_y[k])
                res = splprep(points[k], s=0.0, k=1)
                u_new = np.linspace(res[1].min(), res[1].max(), interp_points_num[k])
                boundary_x[k], boundary_y[k] = splev(u_new, res[0], der=0)
    for index, k in enumerate(boundary_keys):
        if point_num_per_boundary[dataset][index] >= 2.:
            for i in range(len(boundary_x[k]) - 1):
                cv2.line(gt_heatmap[index], (int(boundary_x[k][i]), int(boundary_y[k][i])),
                         (int(boundary_x[k][i+1]), int(boundary_y[k][i+1])), 0)
        else:
            cv2.circle(gt_heatmap[index], (int(boundary_x[k][0]), int(boundary_y[k][0])), 2, 0, -1)
        gt_heatmap[index] = np.uint8(gt_heatmap[index])
        gt_heatmap[index] = cv2.distanceTransform(gt_heatmap[index], cv2.DIST_L2, 5)
        gt_heatmap[index] = np.float32(np.array(gt_heatmap[index]))
        gt_heatmap[index] = gt_heatmap[index].reshape(heatmap_size*heatmap_size)
        (gt_heatmap[index])[(gt_heatmap[index]) < 3. * sigma] = \
            np.exp(-(gt_heatmap[index])[(gt_heatmap[index]) < 3 * sigma] *
                   (gt_heatmap[index])[(gt_heatmap[index]) < 3 * sigma] / 2. * sigma * sigma)
        (gt_heatmap[index])[(gt_heatmap[index]) >= 3. * sigma] = 0.
        gt_heatmap[index] = gt_heatmap[index].reshape([heatmap_size, heatmap_size])
    return np.array(gt_heatmap)


def get_mean_std_color(dataset, split):
    mean = None
    std = None
    if dataset in means_color and split in means_color[dataset]:
        mean = np.float32(np.array(means_color[dataset][split]))
    if dataset in stds_color and split in stds_color[dataset]:
        std = np.float32(np.array(stds_color[dataset][split]))
    return mean, std


def get_mean_std_gray(dataset, split):
    mean = None
    std = None
    if dataset in means_gray and split in means_gray[dataset]:
        mean = means_gray[dataset][split]
    if dataset in stds_gray and split in stds_gray[dataset]:
        std = stds_gray[dataset][split]
    return  np.array(mean),  np.array(std)


def get_item_from(dataset_route, dataset, split, type, annotation, crop_size, RGB, sigma, trans_ratio, rotate_limit,
                  scale_ratio_up, scale_ration_down, scale_horizontal, scale_vertical):
    pic_orig = cv2.imread(dataset_route[dataset] + annotation[-1])
    coord_x = list(map(float, annotation[:2*kp_num[dataset]:2]))
    coord_y = list(map(float, annotation[1:2*kp_num[dataset]:2]))
    coord_xy = np.array(np.float32(annotation[:2*kp_num[dataset]]))
    bbox = np.array(list(map(int, annotation[-7:-3])))

    translation, trans_dir, rotation, scaling, scaling_horizontal, scaling_vertical, flip, gaussian_blur =\
        get_random_transform_param(type, bbox, trans_ratio, rotate_limit, scale_ratio_up, scale_ration_down, scale_horizontal, scale_vertical)

    horizontal_add = (bbox[2] - bbox[0]) * (1 - scaling)
    vertical_add = (bbox[3] - bbox[1]) * (1 - scaling)
    bbox = np.float32(
        [bbox[0] - horizontal_add, bbox[1] - vertical_add, bbox[2] + horizontal_add, bbox[3] + vertical_add])

    horizontal_add = (bbox[2] - bbox[0]) * scaling_horizontal
    vertical_add = (bbox[3] - bbox[1]) * scaling_vertical
    bbox = np.float32([bbox[0] - horizontal_add, bbox[1] - vertical_add, bbox[2] + horizontal_add, bbox[3] + vertical_add])

    position_before = np.float32([[int(bbox[0]) + pow(-1, trans_dir+1)*translation,
                                   int(bbox[1]) + pow(-1, trans_dir//2+1)*translation],
                                  [int(bbox[0]) + pow(-1, trans_dir+1)*translation,
                                   int(bbox[3]) + pow(-1, trans_dir//2+1)*translation],
                                  [int(bbox[2]) + pow(-1, trans_dir+1)*translation,
                                   int(bbox[3]) + pow(-1, trans_dir//2+1)*translation]])
    position_after = np.float32([[0, 0],
                                 [0, crop_size - 1],
                                 [crop_size - 1, crop_size - 1]])
    crop_matrix = cv2.getAffineTransform(position_before, position_after)
    # crop_matrix = np.vstack([crop_matrix, [0, 0, 1]])
    pic_affine_orig = cv2.warpAffine(pic_orig, crop_matrix, (crop_size, crop_size), borderMode=cv2.BORDER_REPLICATE)
    # width_height = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    width_height = (crop_size, crop_size)
    affine_matrix = get_affine_matrix(width_height, rotation, scaling)
    # affine_matrix = np.vstack([affine_matrix, [0, 0, 1]])
    # affine_matrix = np.matmul(crop_matrix, affine_matrix)
    # TODO one transform
    pic_affine_orig = cv2.warpAffine(pic_affine_orig, affine_matrix, (crop_size, crop_size),
                                     borderMode=cv2.BORDER_REPLICATE)
    pic_affine_orig = further_transform(pic_affine_orig, bbox, flip, gaussian_blur) if type in [
        'train'] else pic_affine_orig

    mean_color, std_color = get_mean_std_color(dataset, split)
    mean_gray, std_gray = get_mean_std_gray(dataset, split)

    pic_affine_orig_norm = pic_normalize_color(pic_affine_orig, mean_color, std_color)
    pic_affine_orig_norm = np.moveaxis(pic_affine_orig_norm, -1, 0)
    if not RGB:
        pic_affine = convert_img_to_gray(pic_affine_orig)
        pic_affine = pic_normalize_gray(pic_affine, mean_gray, std_gray)[np.newaxis, ...]
    else:
        pic_affine = pic_affine_orig_norm

    coord_x_cropped, coord_y_cropped = get_cropped_coords(dataset, crop_matrix, coord_x, coord_y, crop_size, flip=flip)
    gt_coords_xy = get_gt_coords(dataset, affine_matrix, coord_x_cropped, coord_y_cropped)

    gt_heatmap = get_gt_heatmap(dataset, gt_coords_xy, crop_size, sigma)

    # heatmap_sum = gt_heatmap[0]
    # for index in range(boundary_num - 1):
    #     heatmap_sum += gt_heatmap[index + 1]
    #
    # for i in range(0, 2*kp_num[dataset], 2):
    #     draw_circle(heatmap_sum, (int(gt_coords_xy[i]/4), int(gt_coords_xy[i+1]/4)))
    #
    # show_img(heatmap_sum)

    pic_affine_orig = np.moveaxis(pic_affine_orig, -1, 0)
    return pic_affine_orig, pic_affine, pic_affine_orig_norm, gt_coords_xy, gt_heatmap, coord_xy, bbox, annotation[-1]


def coords_seq_to_xy(dataset, shapes):
    '''
    [B, x1, y1, x2, y2, ...] -> [B, [x1, y1], [x2, y2], ...]
    :param dataset:
    :param shapes:
    :return:
    '''
    if shapes.ndim == 2:
        if isinstance(shapes, torch.Tensor):
            return torch.reshape(shapes, [-1, kp_num[dataset], 2])
        return shapes.reshape((-1, kp_num[dataset], 2), order='F')
    else:
        if isinstance(shapes, torch.Tensor):
            return torch.reshape(shapes, [-1, kp_num[dataset], 2])
        return shapes.reshape((-1, 2), order='F')



def coords_xy_to_seq(dataset, shapes_xy):
    '''
    [B, [x1, y1], [x2, y2], ...] -> [B, x1, y1, x2, y2, ...]
    :param dataset:
    :param shapes:
    :return:
    '''
    if len(shapes_xy.shape) == 3:
        return shapes_xy.reshape((-1, 2*kp_num[dataset]))
    else:
        return shapes_xy.reshape((-1))