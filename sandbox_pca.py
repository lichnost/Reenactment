from utils.dataload import get_annotations_list
from utils.args import parse_args
from utils.dataset_info import kp_num
import numpy as np
import cv2
from sklearn.decomposition import PCA


def main(arg):
    dataset = arg.dataset
    split = arg.split
    crop_size = arg.crop_size
    annotations = get_annotations_list(arg.dataset_route, dataset, split, crop_size)

    shapes = np.zeros((len(annotations), 2 * kp_num[dataset]))
    for line_index, line in enumerate(annotations):
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
        shapes[line_index, 0:kp_num[dataset]] = list(coord_x_after_crop)
        shapes[line_index, kp_num[dataset]:2 * kp_num[dataset]] = list(coord_y_after_crop)

    mean_shapes = np.mean(shapes, 0)

    faces_pca = PCA(n_components=1)
    components = faces_pca.fit_transform(np.transpose(shapes))
    faces_pca.transform()
    projected = faces_pca.inverse_transform(components)
    print('done')


if __name__ == '__main__':
    arg = parse_args()
    main(arg)