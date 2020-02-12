import torch.utils.data as data
from utils import get_annotations_list, get_item_from
import cv2
from kornia import image_to_tensor, bgr_to_rgb
from .dataset_info import *
import numpy as np
from .dataload import get_gt_coords, further_transform, get_random_transform_param, get_affine_matrix,\
    get_cropped_coords, get_mean_std_color, get_mean_std_gray, pic_normalize_color, convert_img_to_gray, pic_normalize_gray
from utils.pdb import procrustes
from sklearn.decomposition import PCA

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


class ShapeDataset(data.Dataset):

    def __init__(self, arg, dataset, split, pca_components=20):
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
            shapes[0:kp_num[self.dataset], line_index] = list(coord_x_after_crop)
            shapes[kp_num[self.dataset]:2 * kp_num[self.dataset], line_index] = list(coord_y_after_crop)

        aligned_shapes = shapes
        mean_shape = np.mean(aligned_shapes, 1)
        mean_shape_xy = mean_shape.reshape((-1, 2), order='F')
        for i in range(len(aligned_shapes[0])):
            aligned_shape_xy = aligned_shapes[:, i].reshape((-1, 2), order='F')
            tmp_error, tmp_shape, tmp_trans = procrustes(mean_shape_xy, aligned_shape_xy, reflection=False)
            aligned_shapes[:, i] = tmp_shape.reshape((1, -1), order='F')

        mean_shape = np.mean(aligned_shapes, 1)
        mean_shape = mean_shape.repeat(len(aligned_shapes[0])).reshape(-1, len(aligned_shapes[0]))
        aligned_shapes = aligned_shapes - mean_shape

        pca = PCA(n_components=self.pca_components, svd_solver='full')
        pose_params = pca.fit_transform(np.transpose(shapes))

        pca_aligned = PCA(n_components=self.pca_components, svd_solver='full')
        aligned_pose_params = pca_aligned.fit_transform(np.transpose(aligned_shapes))

        self.shapes = np.moveaxis(shapes, -1, 0)
        self.pose_params = pose_params

        self.aligned_shapes = np.moveaxis(aligned_shapes, -1, 0)
        self.aligned_pose_params = aligned_pose_params

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        gt_coords_xy = np.float32(self.shapes[item]).reshape((-1, 2), order='F')
        pose_param = np.float32(self.pose_params[item])

        aligned_coords_xy = np.float32(self.aligned_shapes[item]).reshape((-1, 2), order='F')
        aligned_pose_params = np.float32(self.aligned_pose_params[item])
        return gt_coords_xy, pose_param, aligned_coords_xy, aligned_pose_params


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
                             self.arg.scale_ratio, self.arg.scale_horizontal, self.arg.scale_vertical)


class DecoderDataset(data.Dataset):

    def __init__(self, arg, dataset, split):
        self.arg = arg
        self.dataset = dataset
        self.split = split
        self.type = arg.type
        self.mean_color, self.std_color = get_mean_std_color(self.dataset, self.split)
        self.mean_gray, self.std_gray = get_mean_std_gray(self.dataset, self.split)
        self.list = get_annotations_list(self.arg.dataset_route, dataset, split, arg.crop_size, ispdb=arg.PDB)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        dataset_route, dataset, split, type, annotation, crop_size, RGB, sigma, trans_ratio, rotate_limit, scale_ratio,\
        scale_horizontal, scale_vertical = self.arg.dataset_route, self.dataset, self.split, self.type,\
                                           self.list[item], self.arg.crop_size, self.arg.RGB, self.arg.sigma,\
                                           self.arg.trans_ratio, self.arg.rotate_limit,self.arg.scale_ratio,\
                                           self.arg.scale_horizontal, self.arg.scale_vertical


        pic_orig = cv2.imread(dataset_route[dataset] + annotation[-1])
        coord_x = list(map(float, annotation[:2 * kp_num[dataset]:2]))
        coord_y = list(map(float, annotation[1:2 * kp_num[dataset]:2]))
        bbox = np.array(list(map(int, annotation[-7:-3])))

        translation, trans_dir, rotation, scaling, scaling_horizontal, scaling_vertical, flip, gaussian_blur = get_random_transform_param(
            type, bbox, trans_ratio, rotate_limit, scale_ratio, scale_horizontal, scale_vertical)

        horizontal_add = (bbox[3] - bbox[1]) * scaling_horizontal
        vertical_add = (bbox[2] - bbox[0]) * scaling_vertical
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
        pic_crop_orig = cv2.warpAffine(pic_orig, crop_matrix, (crop_size, crop_size))
        pic_crop_orig = further_transform(pic_crop_orig, bbox, flip, gaussian_blur) if type in [
            'train'] else pic_crop_orig
        affine_matrix = get_affine_matrix(crop_size, rotation, scaling)
        pic_affine_orig = cv2.warpAffine(pic_crop_orig, affine_matrix, (crop_size, crop_size))
        pic_affine_orig = np.float32(cv2.cvtColor(pic_affine_orig, cv2.COLOR_BGR2RGB))

        pic_affine_orig_norm = pic_normalize_color(pic_affine_orig, self.mean_color, self.std_color)
        pic_affine_orig_norm = np.moveaxis(pic_affine_orig_norm, -1, 0)
        if not RGB:
            pic_affine = convert_img_to_gray(pic_affine_orig)
            pic_affine = pic_normalize_gray(pic_affine, self.mean_gray, self.std_gray)[np.newaxis, ...]
        else:
            pic_affine = pic_affine_orig_norm

        coord_x_cropped, coord_y_cropped = get_cropped_coords(dataset, crop_matrix, coord_x, coord_y, crop_size,
                                                              flip=flip)
        gt_coords_xy = get_gt_coords(dataset, affine_matrix, coord_x_cropped, coord_y_cropped)
        return pic_affine, pic_affine_orig_norm, gt_coords_xy