import torch.utils.data as data
from utils import get_annotations_list, get_item_from, show_img
import cv2
from kornia import image_to_tensor, bgr_to_rgb, rgb_to_bgr, normalize, rgb_to_grayscale, tensor_to_image, denormalize
from .dataset_info import *
import numpy as np
from .dataload import get_gt_coords, further_transform, get_random_transform_param, get_affine_matrix,\
    get_cropped_coords, get_mean_std_color, get_mean_std_gray, pic_normalize_color, convert_img_to_gray, pic_normalize_gray,\
    get_gt_heatmap, coords_xy_to_seq, coords_seq_to_xy
from .visual import draw_circle, show_img
from utils.pdb import procrustes
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import torch
import random
import os
import pickle

from flame.FLAME import random_shape_params, random_pose_params, random_expression_params, random_neck_pose_params,\
    random_cam_params, random_scale_params

class OriginalImageDataset(data.Dataset):

    def __init__(self, arg, dataset, split):
        self.arg = arg
        self.dataset = dataset
        self.split = split
        self.list = get_annotations_list(self.arg.dataset_route, dataset, split, arg.crop_size, ispdb=arg.PDB)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        img = cv2.imread(self.arg.dataset_route[self.dataset] + self.list[item][-1])
        img = image_to_tensor(img)
        img = bgr_to_rgb(img)
        return img


class ShapePCADataset(data.Dataset):

    def __init__(self, arg, dataset, split, pca_components=20, trainset_sim=None):
        self.arg = arg
        self.dataset = dataset
        self.split = split
        self.pca_components = pca_components
        self.list = get_annotations_list(self.arg.dataset_route, dataset, split, arg.crop_size, ispdb=arg.PDB)

        self.shapes = None
        self.pose_params = None
        self.aligned_shapes = None
        self.aligned_pose_params = None

        self.init_aligned_shapes(arg.crop_size)

        self.tree = None
        if trainset_sim is not None:
            self.trainset_sim = trainset_sim
            self.tree = KDTree(np.float32(self.trainset_sim.shapes))

    def init_aligned_shapes(self, crop_size):
        shapes = np.zeros((2 * kp_num[self.dataset], len(self.list)))
        for line_index, line in enumerate(self.list):
            coord_x = np.array(list(map(float, line[:2 * kp_num[self.dataset]:2])))
            coord_y = np.array(list(map(float, line[1:2 * kp_num[self.dataset]:2])))
            position_before = np.float32([[int(line[-7]), int(line[-6])],
                                          [int(line[-7]), int(line[-4])],
                                          [int(line[-5]), int(line[-4])]])
            position_after = np.float32([[0, 0],
                                         [0, crop_size - 1],
                                         [crop_size - 1, crop_size - 1]])
            crop_matrix = cv2.getAffineTransform(position_before, position_after)
            coord_x_after_crop = crop_matrix[0][0] * coord_x + crop_matrix[0][1] * coord_y + crop_matrix[0][2]
            coord_y_after_crop = crop_matrix[1][0] * coord_x + crop_matrix[1][1] * coord_y + crop_matrix[1][2]
            shapes[0:2*kp_num[self.dataset]:2, line_index] = list(coord_x_after_crop)
            shapes[1:2*kp_num[self.dataset]:2, line_index] = list(coord_y_after_crop)

        aligned_shapes = shapes
        mean_shape = np.mean(aligned_shapes, 1)
        mean_shape_xy = coords_seq_to_xy(self.dataset, mean_shape)
        for i in range(len(aligned_shapes[0])):
            aligned_shape_xy = coords_seq_to_xy(self.dataset, aligned_shapes[:, i])
            tmp_error, tmp_shape, tmp_trans = procrustes(mean_shape_xy, aligned_shape_xy, reflection=False)
            aligned_shapes[:, i] = tmp_shape.reshape((1, -1), order='F')

        mean_shape = np.mean(aligned_shapes, 1)
        mean_shape = mean_shape.repeat(len(aligned_shapes[0])).reshape(-1, len(aligned_shapes[0]))
        aligned_shapes = aligned_shapes - mean_shape

        shapes = np.moveaxis(shapes, -1, 0)

        # img_show = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        # idx = random.randint(0, shapes.shape[0] - 1)
        # for i in range(0, kp_num[self.dataset] - 1):
        #     draw_circle(img_show, (int(shapes[idx, 2*i]), int(shapes[idx, 2*i+1])))  # red
        #
        # show_img(img_show)

        pca = PCA(n_components=self.pca_components, svd_solver='full')
        pose_params = pca.fit_transform(shapes)

        aligned_shapes = np.moveaxis(aligned_shapes, -1, 0)
        pca_aligned = PCA(n_components=self.pca_components, svd_solver='full')
        aligned_pose_params = pca_aligned.fit_transform(aligned_shapes)

        self.shapes = shapes
        self.pose_params = pose_params

        self.aligned_shapes = aligned_shapes
        self.aligned_pose_params = aligned_pose_params

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        gt_coords_xy = np.float32(self.shapes[item])
        gt_heatmap = get_gt_heatmap(self.dataset, gt_coords_xy.reshape([2*kp_num[self.arg.dataset]]),
                                    self.arg.crop_size, self.arg.sigma)
        pose_param = np.float32(self.pose_params[item])

        aligned_coords_xy = np.float32(self.aligned_shapes[item])
        aligned_pose_params = np.float32(self.aligned_pose_params[item])

        return gt_coords_xy, gt_heatmap, pose_param, aligned_coords_xy, aligned_pose_params

    def get_similars(self, shapes):
        if self.tree is not None:
            _, indexes = self.tree.query(coords_xy_to_seq(self.dataset, shapes))
            return tuple(map(torch.tensor, zip(*[self.trainset_sim[i]
                                                 for i in indexes])))
        return None

class GeneralDataset(data.Dataset):

    def __init__(self, arg, dataset, split):
        self.arg = arg
        self.dataset = dataset
        self.split = split
        self.type = arg.type
        self.list = get_annotations_list(self.arg.dataset_route, dataset, split, arg.crop_size, ispdb=arg.PDB)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        return get_item_from(self.arg.dataset_route, self.dataset, self.split, self.type, self.list[item], self.arg.crop_size,
                             self.arg.RGB, self.arg.sigma, self.arg.trans_ratio, self.arg.rotate_limit,
                             self.arg.scale_ratio_up, self.arg.scale_ratio_down, self.arg.scale_horizontal, self.arg.scale_vertical)


class DecoderDataset(data.Dataset):

    def __init__(self, arg, dataset, split):
        self.arg = arg
        self.dataset = dataset
        self.split = split
        self.type = arg.type
        self.mean_color, self.std_color = get_mean_std_color(self.dataset, self.split)
        self.mean_gray, self.std_gray = get_mean_std_gray(self.dataset, self.split)
        self.list = get_annotations_list(self.arg.dataset_route, dataset, split, arg.crop_size, ispdb=arg.PDB)
        if len(arg.dataset_indexes) > 0:
            self.list = [self.list[i] for i in arg.dataset_indexes]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        dataset_route, dataset, split, type, annotation, crop_size, RGB, sigma, trans_ratio, rotate_limit,\
        scale_ratio_up, scale_ratio_down, scale_horizontal, scale_vertical =\
            self.arg.dataset_route, self.dataset, self.split, self.type,\
            self.list[item], self.arg.crop_size, self.arg.RGB, self.arg.sigma,\
            self.arg.trans_ratio, self.arg.rotate_limit, self.arg.scale_ratio_up, self.arg.scale_ratio_down,\
            self.arg.scale_horizontal, self.arg.scale_vertical


        pic_orig = cv2.imread(dataset_route[dataset] + annotation[-1])
        coord_x = list(map(float, annotation[:2 * kp_num[dataset]:2]))
        coord_y = list(map(float, annotation[1:2 * kp_num[dataset]:2]))
        bbox = np.array(list(map(int, annotation[-7:-3])))

        translation, trans_dir, rotation, scaling, scaling_horizontal, scaling_vertical, flip, gaussian_blur = get_random_transform_param(
            type, bbox, trans_ratio, rotate_limit, scale_ratio_up, scale_ratio_down, scale_horizontal, scale_vertical, flip=False, gaussian=False)

        horizontal_add = (bbox[2] - bbox[0]) * (1 - scaling)
        vertical_add = (bbox[3] - bbox[1]) * (1 - scaling)
        bbox = np.float32(
            [bbox[0] - horizontal_add, bbox[1] - vertical_add, bbox[2] + horizontal_add, bbox[3] + vertical_add])

        horizontal_add = (bbox[2] - bbox[0]) * scaling_horizontal
        vertical_add = (bbox[3] - bbox[1]) * scaling_vertical
        bbox = np.float32(
            [bbox[0] - horizontal_add, bbox[1] - vertical_add, bbox[2] + horizontal_add, bbox[3] + vertical_add])

        position_before = np.float32([[int(bbox[0]) + pow(-1, trans_dir + 1) * translation,
                                       int(bbox[1]) + pow(-1, trans_dir // 2 + 1) * translation],
                                      [int(bbox[0]) + pow(-1, trans_dir + 1) * translation,
                                       int(bbox[3]) + pow(-1, trans_dir // 2 + 1) * translation],
                                      [int(bbox[2]) + pow(-1, trans_dir + 1) * translation,
                                       int(bbox[3]) + pow(-1, trans_dir // 2 + 1) * translation]])
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
        pic_affine_orig = cv2.warpAffine(pic_affine_orig, affine_matrix, (crop_size, crop_size), borderMode=cv2.BORDER_REPLICATE)
        pic_affine_orig = further_transform(pic_affine_orig, bbox, flip, gaussian_blur) if type in ['train'] else pic_affine_orig

        # show_img(pic_affine_orig, wait=0, keep=False)
        pic_affine_orig = bgr_to_rgb(image_to_tensor(pic_affine_orig))

        pic_affine_orig_norm = normalize(pic_affine_orig, torch.from_numpy(self.mean_color), torch.from_numpy(self.std_color))
        if not RGB:
            pic_affine = convert_img_to_gray(pic_affine_orig)
            pic_affine = normalize(pic_affine, self.mean_gray, self.std_gray)
        else:
            pic_affine = pic_affine_orig_norm

        coord_x_cropped, coord_y_cropped = get_cropped_coords(dataset, crop_matrix, coord_x, coord_y, crop_size,
                                                              flip=flip)
        gt_coords_xy = get_gt_coords(dataset, affine_matrix, coord_x_cropped, coord_y_cropped)

        return pic_affine, pic_affine_orig_norm, gt_coords_xy


class ShapeFlameDataset(data.Dataset):

    def __init__(self, arg, dataset, split):
        self.arg = arg
        self.dataset = dataset
        self.split = split
        self.mean = arg.flame_dataset_mean_shape

        self.list = get_annotations_list(self.arg.dataset_route, dataset, split, arg.crop_size, ispdb=arg.PDB)

        self.shape_params = []
        self.pose_params = []
        self.expression_params = []
        self.cam_params = []
        self.mean_shape_params = None
        self.load_flame_params()

    def load_flame_params(self):
        for i, item in enumerate(self.list):
            path = os.path.join(self.arg.dataset_route[self.dataset], 'FLAME', self.split, 'params', os.path.splitext(os.path.split(item[-1])[-1])[0]) + os.path.extsep + 'npy'
            params = np.load(path, allow_pickle=True, encoding='latin1').item()

            self.shape_params.append(np.float32(params['shape']))
            self.pose_params.append(np.float32(params['pose'][:6]))
            self.expression_params.append(np.float32(np.append(params['expression'], 0.0)))
            self.cam_params.append(np.float32(params['cam']))

        self.mean_shape_params = np.mean(np.array(self.shape_params), axis=0)
        if self.arg.flame_dataset_mean_shape:
            self.shape_params = [self.mean_shape_params] * len(self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        scale = 1
        if self.arg.flame_random_params:
            shape = random_shape_params()
            pose = random_pose_params()
            neck_pose = random_neck_pose_params()
            expression = random_expression_params()
            cam = random_cam_params()
            scale = random_scale_params()
        else:
            shape = self.shape_params[item]
            pose = self.pose_params[item]
            neck_pose = np.zeros((3), dtype=np.float32)
            expression = self.expression_params[item]
            cam = self.cam_params[item]

        return self.list[item][-1], shape, pose, neck_pose, expression, cam, scale