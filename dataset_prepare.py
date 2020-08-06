import os

import cv2
import torch
import dlib
from tqdm import tqdm

from models import Estimator, Regressor
from utils import *
from utils.args import parse_args
from utils.dataset_info import kp_num
from utils.dataload import get_annos_path

def main(arg):
    devices = get_devices_list(arg)

    face_detector = dlib.cnn_face_detection_model_v1(arg.dlib_face_detector_path)
    shape_predictor = None
    # shape_predictor = dlib.shape_predictor(arg.dlib_shape_predictor_path)

    points_num = kp_num[arg.dataset]

    print('Creating networks ...')
    estimator = create_model_estimator(arg, devices, eval=True)
    estimator.eval()

    regressor = create_model_regressor(arg, devices, eval=True)
    regressor.eval()

    path_split = os.path.join(arg.dataset_route[arg.dataset], arg.split)
    annotation_lines = []
    if arg.video_path is None:
        for root, dirs, files in os.walk(path_split):
            for file in tqdm(files):
                # print(file)
                file_path = os.path.join(root, file)
                img_original = cv2.imread(file_path)
                annotation_lines = annotation_lines + detect_annotations(arg, img_original, estimator, regressor,
                                                                         face_detector, points_num, file, devices)
    else:
        if not os.path.exists(path_split):
            os.mkdir(path_split)

        capture = cv2.VideoCapture(arg.video_path)
        image_name = os.path.splitext(os.path.split(arg.video_path)[-1])[0]

        frame_idx = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            if frame is None:
                break

            image_name_indexed = image_name + '_' + str(frame_idx) + '.jpg'
            image_path = os.path.join(arg.dataset_route[arg.dataset], arg.split, image_name_indexed)
            detected_annotations = detect_annotations(arg, frame, estimator, regressor, face_detector, points_num,
                                                      image_name_indexed, devices)
            annotation_lines = annotation_lines + detected_annotations
            if len(detected_annotations) > 0:
                cv2.imwrite(image_path, frame)
                frame_idx += 1

    annotation_path = get_annos_path(arg.dataset_route[arg.dataset], arg.dataset, arg.split)
    if os.path.exists(annotation_path) and os.path.isfile(annotation_path):
        os.remove(annotation_path)

    with open(annotation_path, 'a') as annotation_file:
        first = True
        for line in annotation_lines:
            if not first:
                annotation_file.write('\n')
            else:
                first = False
            line = list(map(lambda item: str(item), line))
            annotation_file.write(' '.join(line))


def detect_annotations(arg, img_original, estimator, regressor, face_detector, points_num, file, devices):
    annotation_lines = []

    img_for_detect = img_original

    pixels = img_original.shape[0] * img_original.shape[1]
    pixels_max = 3000000
    scale_factor = 1
    if pixels > pixels_max: # 2MP gpu memory overflow
        scale_factor = pixels_max / pixels
        img_for_detect = cv2.resize(img_original,
                                    (int(img_original.shape[0] * scale_factor),
                                     int(img_original.shape[1] * scale_factor)))

    faces = face_detector(img_for_detect, 0)
    for face in faces:
        rec_list = face.rect

        # bottom = rec_list.bottom()
        # top = rec_list.top()
        # right = rec_list.right()
        # left = rec_list.left()

        bottom = rec_list.bottom() * (1/scale_factor)
        top = rec_list.top() * (1/scale_factor)
        right = rec_list.right() * (1/scale_factor)
        left = rec_list.left() * (1/scale_factor)

        height = bottom - top
        width = right - left
        # detect coords firstly to normalize face position
        bbox = [
            int(left - arg.scale_ratio * width),
            int(top - arg.scale_ratio * height),
            int(right + arg.scale_ratio * width),
            int(bottom + arg.scale_ratio * height)
        ]

        coords, _, inv_crop_matrix, heatmap = detect_coords(arg, img_original, bbox, arg.crop_size, estimator, regressor, devices)

        annotation = []

        for index in range(points_num):
            x, y = coords[2 * index], coords[2 * index + 1]
            (x_t, y_t) = coord_transform((x, y), inv_crop_matrix)
            coords[2 * index], coords[2 * index + 1] = x_t, y_t

            annotation.append(x_t)
            annotation.append(y_t)

            # cv2.circle(img_original, (int(x_t), int(y_t)), 2, (0, 0, 255), -1)

        bbox = normalized_bbox(coords, arg.dataset, face_size=arg.normalize_face_size, top_shift=arg.normalize_top_shift)
        bbox = [
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[2]),
            int(bbox[3])
        ]

        # bbox
        annotation.append(bbox[0])
        annotation.append(bbox[1])
        annotation.append(bbox[2])
        annotation.append(bbox[3])

        # image size
        annotation.append(img_original.shape[0])
        annotation.append(img_original.shape[1])

        # path
        annotation.append(os.path.join(arg.split, file))
        annotation_lines.append(annotation)

        if arg.save_heatmaps:
            heatmaps_data = os.path.join(arg.dataset_route[arg.dataset], 'heatmaps', arg.split, 'data')
            heatmaps_images = os.path.join(arg.dataset_route[arg.dataset], 'heatmaps', arg.split, 'images')
            if not os.path.exists(heatmaps_data):
                os.makedirs(heatmaps_data)
            if not os.path.exists(heatmaps_images):
                os.makedirs(heatmaps_images)
            heatmaps_file = os.path.splitext(file)[0]
            np.save(os.path.join(heatmaps_data, heatmaps_file + '.npy'), heatmap.squeeze().detach().cpu().numpy())
            cv2.imwrite(os.path.join(heatmaps_images, heatmaps_file + '.jpg'), np.moveaxis(get_heatmap_gray(heatmap, True).detach().cpu().numpy(), 0, -1))

        # cv2.rectangle(img_original, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # cv2.imshow('original', img_original)
        # cv2.waitKey()
        # cv2.destroyWindow('original')

    return annotation_lines

if __name__ == '__main__':
    arg = parse_args()
    arg.GAN = False
    arg.scale_ratio = 0.5
    main(arg)
