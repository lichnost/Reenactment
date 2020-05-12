import copy
import os

import dlib
import torch.nn.functional as F
from kornia.color import denormalize
from kornia.geometry.transform import warp_affine

from utils import *
from utils.args import parse_args

import matplotlib.pyplot as plt


def main(arg):
    devices = get_devices_list(arg)

    # load network
    print('*****  ' + arg.dataset + ' boundary Model Evaluating  *****')
    print('Loading network ...')
    estimator = create_model_estimator(arg, devices, eval=True)
    estimator.eval()

    if arg.normalized_bbox:
        regressor = create_model_regressor(arg, devices, eval=True)
        regressor.eval()

    transformer = create_model_transformer_a2b(arg, devices, eval=True)
    transformer.eval()

    edge = create_model_edge(arg, devices, eval=True)
    edge.eval()

    decoder = create_model_decoder(arg, devices, eval=True)
    decoder.eval()

    print('Loading network done!\nStart testing ...')

    mean = torch.FloatTensor(means_color[arg.eval_dataset_decoder][arg.eval_split_decoder])
    std = torch.FloatTensor(stds_color[arg.eval_dataset_decoder][arg.eval_split_decoder])

    if arg.cuda:
        mean = mean.cuda(device=devices[0])
        std = std.cuda(device=devices[0])

    if arg.eval_video_path is not None:
        cap = cv2.VideoCapture(arg.eval_video_path)
    else:
        cap = cv2.VideoCapture(0)

    # detect face and facial landmark
    while cap.isOpened():      # isOpened()  Detect if the camera is on
        ret, img = cap.read()  # Save the image information obtained by the camera to the img variable
        if ret is True:        # If the camera reads the image successfully

            # if arg.eval_visual:
            show_img(img, 'source', wait=1, keep=True)

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

                        if arg.normalized_bbox:
                            coords, crop_matrix, inv_crop_matrix, heatmaps = detect_coords(arg, img, bbox, arg.crop_size, estimator,
                                                                    regressor, devices)
                            for index in range(kp_num[arg.dataset]):
                                x, y = coords[2 * index], coords[2 * index + 1]
                                (x_t, y_t) = coord_transform((x, y), inv_crop_matrix)
                                coords[2 * index], coords[2 * index + 1] = x_t, y_t


                            inv_crop_matrix = torch.tensor(np.float32(inv_crop_matrix[np.newaxis, :, :]))
                            if arg.cuda:
                                inv_crop_matrix = inv_crop_matrix.cuda(device=devices[0])
                            heatmaps = F.interpolate(heatmaps, arg.crop_size, mode='bicubic')
                            heatmaps = warp_affine(heatmaps, inv_crop_matrix, (img.shape[0], img.shape[1]), padding_mode='border')

                            norm_bbox = normalized_bbox(coords, arg.dataset, face_size=arg.normalize_face_size, top_shift=arg.normalize_top_shift)
                            position_before = np.float32([
                                [int(norm_bbox[0]), int(norm_bbox[1])],
                                [int(norm_bbox[0]), int(norm_bbox[3])],
                                [int(norm_bbox[2]), int(norm_bbox[3])]
                            ])
                            position_after = np.float32([
                                [0, 0],
                                [0, 63],
                                [63, 63]
                            ])

                            crop_matrix = cv2.getAffineTransform(position_before, position_after)
                            crop_matrix = torch.tensor(np.float32(crop_matrix[np.newaxis, :, :]))
                            if arg.cuda:
                                crop_matrix = crop_matrix.cuda(device=devices[0])
                            heatmaps = warp_affine(heatmaps, crop_matrix, (64, 64), padding_mode='border')

                        else:
                            position_before = np.float32([
                                [int(bbox[0]), int(bbox[1])],
                                [int(bbox[0]), int(bbox[3])],
                                [int(bbox[2]), int(bbox[3])]
                            ])
                            position_after = np.float32([
                                [0, 0],
                                [0, arg.crop_size - 1],
                                [arg.crop_size - 1, arg.crop_size - 1]
                            ])
                            crop_matrix = cv2.getAffineTransform(position_before, position_after)
                            face_img = cv2.warpAffine(img, crop_matrix, (arg.crop_size, arg.crop_size))
                            face_gray = convert_img_to_gray(face_img)
                            face_norm = pic_normalize_gray(face_gray)

                            input_face = torch.Tensor(face_norm)
                            input_face = input_face.unsqueeze(0).unsqueeze(0)
                            if arg.cuda:
                                input_face = input_face.cuda(device=devices[0])

                            heatmaps_orig = estimator(input_face)
                            heatmaps = heatmaps_orig[-1]

                        # heatmaps = transformer(heatmaps)

                        # heatmaps = F.interpolate(heatmaps, arg.crop_size, mode='bicubic')
                        heatmaps = edge(heatmaps)
                        # heatmaps[heatmaps < arg.boundary_cutoff_lambda * heatmaps.max()] = 0

                        fake_image_norm = decoder(heatmaps).detach()
                        fake_image = denormalize(fake_image_norm, mean, std).cpu().squeeze().numpy()
                        fake_image = np.uint8(np.moveaxis(fake_image, 0, -1))
                        fake_image = cv2.cvtColor(fake_image, cv2.COLOR_RGB2BGR)

                        if arg.eval_visual:
                            show_img(fake_image, 'target', wait=1, keep=True)

                            heatmap_show = get_heatmap_gray(heatmaps).detach().cpu().numpy()
                            heatmap_show = (
                                    255 - np.uint8(255 * (heatmap_show - np.min(heatmap_show)) / np.ptp(heatmap_show)))
                            heatmap_show = np.moveaxis(heatmap_show, 0, -1)
                            heatmap_show = cv2.resize(heatmap_show, (256, 256))

                            show_img(heatmap_show, 'heatmap', wait=1, keep=True)


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