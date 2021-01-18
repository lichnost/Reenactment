import os

from utils import *
from utils.args import parse_args
import numpy as np
from sklearn.decomposition import PCA


def main(arg):
    annotations = get_annotations_list(arg.dataset_route, arg.dataset, arg.split, arg.crop_size, ispdb=arg.PDB)

    landmarks = np.zeros((len(annotations), 2 * kp_num[arg.dataset]))
    for line_index, line in enumerate(annotations):
        coord_x = np.array(list(map(float, line[:2 * kp_num[arg.dataset]:2])))
        coord_y = np.array(list(map(float, line[1:2 * kp_num[arg.dataset]:2])))
        position_before = np.float32([[int(line[-7]), int(line[-6])],
                                      [int(line[-7]), int(line[-4])],
                                      [int(line[-5]), int(line[-4])]])
        position_after = np.float32([[0, 0],
                                     [0, arg.crop_size - 1],
                                     [arg.crop_size - 1, arg.crop_size - 1]])
        crop_matrix = cv2.getAffineTransform(position_before, position_after)
        coord_x_after_crop = crop_matrix[0][0] * coord_x + crop_matrix[0][1] * coord_y + crop_matrix[0][2]
        coord_y_after_crop = crop_matrix[1][0] * coord_x + crop_matrix[1][1] * coord_y + crop_matrix[1][2]
        landmarks[line_index, 0:2 * kp_num[arg.dataset]:2] = list(coord_x_after_crop)
        landmarks[line_index, 1:2 * kp_num[arg.dataset]:2] = list(coord_y_after_crop)
    landmarks = coords_seq_to_xy(arg.dataset, landmarks)

    mean = np.mean(landmarks.reshape(-1, landmarks.shape[-1]), axis=0)
    landmarks = landmarks - mean

    landmarks_transposed = landmarks.transpose((0, 2, 1))

    reshaped = landmarks_transposed.reshape(landmarks.shape[0], -1)

    pca = PCA(n_components=arg.pca_components)
    pca.fit(reshaped)
    components = pca.components_ * pca.singular_values_.reshape(-1, 1)
    # idx 0: mean, 1...: components
    result = np.array([pca.mean_] + list(components)).reshape(
            components.shape[0] + 1,
            *landmarks_transposed.shape[1:]).transpose(0, 2, 1)
    directory = os.path.join(arg.dataset_route[arg.dataset], 'pca', arg.split)
    if not os.path.exists(directory) and not os.path.isdir(directory):
        os.makedirs(directory)
    np.save(os.path.join(directory, arg.split) + '.npy', result)

if __name__ == '__main__':
    arg = parse_args()
    arg.GAN = False
    arg.scale_ratio = 0.5
    main(arg)
