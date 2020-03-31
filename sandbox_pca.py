import torch.nn as nn
import tqdm
import numpy as np
from sklearn.decomposition import PCA

from utils import *
from utils.args import parse_args
from utils.dataset import ShapeDataset
import cv2
from utils.pdb import procrustes


def init_aligned_shapes(dataset, shape_list, crop_size):
    shapes = np.zeros((2 * kp_num[dataset], len(shape_list)))
    for line_index, line in enumerate(shape_list):
        coord_x = np.array(list(map(float, line[:2 * kp_num[dataset]:2])))
        coord_y = np.array(list(map(float, line[1:2 * kp_num[dataset]:2])))
        position_before = np.float32([[int(line[-7]), int(line[-6])],
                                      [int(line[-7]), int(line[-4])],
                                      [int(line[-5]), int(line[-4])]])
        position_after = np.float32([[0, 0],
                                     [0, crop_size - 1],
                                     [crop_size - 1, crop_size - 1]])
        crop_matrix = cv2.getAffineTransform(position_before, position_after)
        coord_x_after_crop = crop_matrix[0][0] * coord_x + crop_matrix[0][1] * coord_y + crop_matrix[0][2]
        coord_y_after_crop = crop_matrix[1][0] * coord_x + crop_matrix[1][1] * coord_y + crop_matrix[1][2]
        shapes[0:kp_num[dataset], line_index] = list(coord_x_after_crop)
        shapes[kp_num[dataset]:2 * kp_num[dataset], line_index] = list(coord_y_after_crop)

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

    return np.transpose(shapes), np.transpose(aligned_shapes)

def main(arg):
    list = get_annotations_list(arg.dataset_route, arg.dataset, arg.split, arg.crop_size, ispdb=arg.PDB)
    list_src = get_annotations_list(arg.dataset_route, arg.dataset, arg.split_source, arg.crop_size, ispdb=arg.PDB)

    shapes, aligned_shapes = init_aligned_shapes(arg.dataset, list, arg.crop_size)
    shapes_src, aligned_shapes_src = init_aligned_shapes(arg.dataset, list_src, arg.crop_size)

    shapes_xy = np.zeros(shapes.shape)
    shapes_xy[:, :shapes.shape[1]:2] = shapes[:, :kp_num[arg.dataset]]
    shapes_xy[:, 1:shapes.shape[1]:2] = shapes[:, kp_num[arg.dataset]:]

    pca = PCA(n_components=arg.pca_components, svd_solver='full')
    pca.fit(shapes)

    pca_src = PCA(n_components=arg.pca_components, svd_solver='full')
    pca_src.fit(shapes_src)

    pose_params = pca.transform(shapes)
    pose_params_mean = pose_params.copy()
    pose_params_mean[:, :] = 0

    mean_shapes = pca.inverse_transform(pose_params_mean)
    mean_shapes_xy = np.zeros(mean_shapes.shape)
    mean_shapes_xy[:, :mean_shapes.shape[1]:2] = mean_shapes[:, :kp_num[arg.dataset]]
    mean_shapes_xy[:, 1:mean_shapes.shape[1]:2] = mean_shapes[:, kp_num[arg.dataset]:]

    img_mean = np.zeros((arg.crop_size, arg.crop_size, 3), dtype=np.uint8)
    items = 20
    for x in range(0, items):
        index = random.randint(0, shapes.shape[0] - 1)
        img = np.zeros((arg.crop_size, arg.crop_size, 3), dtype=np.uint8)
        for i in range(0, kp_num[arg.dataset]):
            xy_mean = (int(mean_shapes[index, i]), int(mean_shapes[index, kp_num[arg.dataset] + i]))
            draw_circle(img_mean, xy_mean, color=(0, 255, 0))  # green
            draw_text(img_mean, str(i), xy_mean, color=(0, 0, 255), scale=0.25)

            xy = (int(shapes[index, i]), int(shapes[index, kp_num[arg.dataset] + i]))
            draw_circle(img, xy, color=(0, 255, 0)) # green

        landmarks = orientation_landmarks(arg.dataset, shapes_xy[index])
        model = orientation_landmarks(arg.dataset, mean_shapes_xy[index])
        for i in range(0, len(landmarks)):
            xy = landmarks[i]
            draw_circle(img, (int(xy[0]), int(xy[1])), color=(255, 0, 0))  # red

        orientation = face_orientation((arg.crop_size, arg.crop_size), landmarks, model)
        draw_orientation(img, orientation)
        modelpts = orientation[1]
        # modelpts[:, 0] += orientation[3][0]
        # modelpts[:, 1] += orientation[3][1]
        for i in range(0, modelpts.shape[0]):
            xy = modelpts[i].astype(int)
            draw_circle(img, tuple(xy), color=(0, 0, 255))  # red

        print(orientation)
        show_img(img_mean, 'mean', keep=True)
        show_img(img)



    # pca_src = PCA(n_components=arg.pca_components, svd_solver='full')
    # _ = pca_src.fit(shapes_src)
    # restored_shapes_src = pca_src.inverse_transform(pose_params)
    #
    # items = 20
    # for x in range(0, items):
    #     index = random.randint(0, shapes.shape[0]-1)
    #     img = np.zeros((arg.crop_size, arg.crop_size, 3), dtype=np.uint8)
    #     for i in range(0, kp_num[arg.dataset]-1):
    #         draw_circle(img, (int(shapes[index, i]), int(shapes[index, kp_num[arg.dataset]+ i]))) # red
    #         draw_circle(img, (int(restored_shapes_src[index, i]), int(restored_shapes_src[index, kp_num[arg.dataset] + i])), color=(255, 0, 0)) # blue
    #
    #     show_img(img)

    print('done')


if __name__ == '__main__':
    arg = parse_args()
    main(arg)