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
                annotation_lines = annotation_lines + detect_annotations(arg, img_original, estimator, regressor, face_detector, shape_predictor, points_num, file)
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
            detected_annotations = detect_annotations(arg, frame, estimator, regressor, face_detector, shape_predictor, points_num, image_name_indexed)
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


def detect_annotations(arg, img_original, estimator, regressor, face_detector, shape_predictor, points_num, file):
    annotation_lines = []

    faces = face_detector(img_original, 1)
    for face in faces:
        rec_list = face.rect
        height = rec_list.bottom() - rec_list.top()
        width = rec_list.right() - rec_list.left()
        bbox = [
            int(rec_list.left() - arg.scale_ratio * width),
            int(rec_list.top() - arg.scale_ratio * height),
            int(rec_list.right() + arg.scale_ratio * width),
            int(rec_list.bottom() + arg.scale_ratio * height)
        ]
        position_before = np.float32([
            [int(bbox[0]), int(bbox[1])],
            [int(bbox[0]), int(bbox[3])],
            [int(bbox[2]), int(bbox[3])]
        ])
        position_after = np.float32([[0, 0],
                                     [0, arg.crop_size - 1],
                                     [arg.crop_size - 1, arg.crop_size - 1]])
        crop_matrix = cv2.getAffineTransform(position_before, position_after)
        inv_crop_matrix = cv2.invertAffineTransform(crop_matrix)

        # cv2.imshow('img_original', img_original[rec_list.top():rec_list.top()+height, rec_list.left():rec_list.left()+width])
        # cv2.waitKey()
        # cv2.destroyWindow('img_original')

        img_color = cv2.warpAffine(img_original, crop_matrix, (arg.crop_size, arg.crop_size))

        # cv2.imshow('img', img)
        # cv2.waitKey()
        # cv2.destroyWindow('img')
        img_color = np.float32(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        img = convert_img_to_gray(img_color)
        img = pic_normalize_gray(img)

        input = torch.Tensor(img)
        input = input.unsqueeze(0)
        input = input.unsqueeze(0).cuda()

        heatmap = estimator(input)[-1]
        coords = regressor(input, heatmap).detach().cpu().squeeze().numpy()  # output x and y: points_num*2
        # coords_5p = shape_predictor(img_color, dlib.rectangle(0, 0, arg.crop_size - 1, arg.crop_size - 1))

        # for index in range(coords_5p.num_parts):
        #     point = coords_5p.part(index)
        #     x, y = point.x, point.y
        #     (x_t, y_t) = coord_transform((x, y), inv_crop_matrix)
        #     # annotation.append(x_t)
        #     # annotation.append(y_t)
        #
        #     # cv2.circle(img_original, (int(x_t), int(y_t)), 2, (0, 0, 255), -1)
        #     # cv2.circle(img_color, (int(x), int(y)), 2, (0, 0, 255), -1)

        annotation = []
        for index in range(points_num):
            x, y = coords[2 * index], coords[2 * index + 1]
            (x_t, y_t) = coord_transform((x, y), inv_crop_matrix)
            annotation.append(x_t)
            annotation.append(y_t)

            # cv2.circle(img_original, (int(x_t), int(y_t)), 2, (0, 0, 255), -1)
            # cv2.circle(img_color, (int(x), int(y)), 2, (0, 255, 0), -1)

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

        # cv2.imshow('original', img_original)
        # cv2.waitKey()
        # cv2.destroyWindow('original')
        # cv2.imshow('img_color', img_color)
        # cv2.waitKey(200)
        # cv2.destroyWindow('img_color')

    return annotation_lines

if __name__ == '__main__':
    arg = parse_args()
    arg.GAN = False
    arg.scale_ratio = 0.5
    main(arg)
