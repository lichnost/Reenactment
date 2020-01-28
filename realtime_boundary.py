import os
import cv2
import dlib
import time
import copy
import numpy as np
from models import *
from utils import *
from utils.args import parse_args
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline as spline


def main(arg):

    devices = get_devices_list(arg)

    # load network
    print('*****  ' + arg.dataset + ' boundary Model Evaluating  *****')
    print('Loading network ...')
    estimator = create_model_estimator(arg, devices, eval=True)
    estimator.eval()

    regressor = create_model_regressor(arg, devices, eval=True)
    regressor.eval()
    print('Loading network done!\nStart testing ...')

    # detect face and facial landmark
    cap = cv2.VideoCapture(0)
    face_keypoint_coords = []

    while cap.isOpened():      # isOpened()  Detect if the camera is on
        ret, img = cap.read()  # Save the image information obtained by the camera to the img variable
        if ret is True:        # If the camera reads the image successfully
            # cv2.imshow('Image', img)
            k = cv2.waitKey(1)
            if arg.realtime or k == ord('c') or k == ord('C'):

                face_detector = dlib.cnn_face_detection_model_v1(arg.dlib_face_detector_path)
                rec = face_detector(img, 1)

                with torch.no_grad():
                    for face_i in range(len(rec)):

                        rec_list = rec.pop().rect
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
                        position_after = np.float32([
                            [0, 0],
                            [0, 255],
                            [255, 255]
                        ])
                        crop_matrix = cv2.getAffineTransform(position_before, position_after)
                        face_img = cv2.warpAffine(img, crop_matrix, (256, 256))
                        face_gray = convert_img_to_gray(face_img)
                        face_norm = pic_normalize_gray(face_gray)

                        input_face = torch.Tensor(face_norm)
                        input_face = input_face.unsqueeze(0)
                        input_face = input_face.unsqueeze(0).cuda(device=devices[0])

                        pred_heatmaps = estimator(input_face)
                        pred_coords = regressor(input_face, pred_heatmaps[-1].detach()).detach().cpu().squeeze().numpy()

                        # for kp_index in range(kp_num[use_dataset]):
                        #     cv2.circle(
                        #         face_img,
                        #         (int(pred_coords[2 * kp_index]), int(pred_coords[2 * kp_index + 1])),
                        #         2,
                        #         (0, 0, 255),
                        #         -1
                        #     )
                        # show_img(face_img, 'face_small_keypoint'+str(face_i), 500, 650, keep=True)
                        # if save_imgs:
                        #     cv2.imwrite('./pics/face_' + t + '_0.png', face_img)
                        #
                        # heatmaps = F.interpolate(
                        #     pred_heatmaps[-1],
                        #     scale_factor=4,
                        #     mode='bilinear',
                        #     align_corners=True
                        # )
                        # heatmaps = heatmaps.squeeze(0).detach().cpu().numpy()
                        # heatmaps_sum = heatmaps[0]
                        # for heatmaps_index in range(boundary_num - 1):
                        #     heatmaps_sum += heatmaps[heatmaps_index + 1]

                        # plt.axis('off')
                        # plt.imshow(heatmaps_sum, interpolation='nearest', vmax=1., vmin=0.)
                        #
                        # fig = plt.gcf()
                        # fig.set_size_inches(2.56 / 3, 2.56 / 3)
                        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                        # plt.margins(0, 0)
                        #
                        # fig.canvas.draw()
                        # hm = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                        # hm = hm.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        # hm = cv2.resize(hm, face_img.shape[:2])
                        # syn = cv2.addWeighted(face_img, 0.4, hm, 0.6, 0)
                        # show_img(syn, 'face_small_boundary'+str(face_i), 900, 650, keep=True)

                        pred_coords_copy = copy.deepcopy(pred_coords)
                        for i in range(kp_num[arg.dataset]):
                            pred_coords_copy[2 * i] = \
                                bbox[0] + pred_coords_copy[2 * i] / 255 * (bbox[2] - bbox[0])
                            pred_coords_copy[2 * i + 1] = bbox[1] + pred_coords_copy[2 * i + 1] / 255 * (
                                        bbox[3] - bbox[1])
                        face_keypoint_coords.append(pred_coords_copy)

                if len(face_keypoint_coords) != 0:
                    for face_id, coords in enumerate(face_keypoint_coords):
                        for kp_index in range(kp_num[arg.dataset]):
                            cv2.circle(
                                img,
                                (int(coords[2 * kp_index]), int(coords[2 * kp_index + 1])),
                                2,
                                (0, 0, 255),
                                -1
                            )
                    show_img(img, 'face_whole', wait=1, keep=True)
                    face_keypoint_coords = []

            if k == ord('q') or k == ord('Q'):
                break

    print('QUIT.')
    cap.release()              # 关闭摄像头
    cv2.destroyAllWindows()


if __name__ == '__main__':
    arg = parse_args()

    arg.realtime = True
    arg.scale_ratio = 0.5

    main(arg)