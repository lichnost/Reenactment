import copy
import os

import dlib
import torch.nn.functional as F
from kornia.color import denormalize

from utils import *
from utils.args import parse_args


def main(arg):
    devices = get_devices_list(arg)

    # load network
    print('*****  ' + arg.dataset + ' boundary Model Evaluating  *****')
    print('Loading network ...')
    estimator = create_model_estimator(arg, devices, eval=True)
    estimator.eval()

    decoder = create_model_decoder(arg, devices, eval=True)
    decoder.eval()

    print('Loading network done!\nStart testing ...')

    mean = torch.FloatTensor(means_color[arg.eval_dataset_decoder][arg.eval_split_decoder])
    std = torch.FloatTensor(stds_color[arg.eval_dataset_decoder][arg.eval_split_decoder])

    if arg.cuda:
        mean = mean.cuda(device=devices[0])
        std = std.cuda(device=devices[0])

    # detect face and facial landmark
    cap = cv2.VideoCapture(0)

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
                        input_face = input_face.unsqueeze(0).unsqueeze(0)
                        if arg.cuda:
                            input_face = input_face.cuda(device=devices[0])

                        heatmaps_orig = estimator(input_face)[-1]
                        heatmaps = F.interpolate(heatmaps_orig, 256, mode='bicubic')
                        fake_image_norm = decoder(heatmaps).detach()
                        fake_image = denormalize(fake_image_norm, mean, std).cpu().squeeze().numpy()
                        fake_image = np.uint8(np.moveaxis(fake_image, 0, -1))
                        fake_image = cv2.cvtColor(fake_image, cv2.COLOR_RGB2BGR)
                        show_img(img, 'source', wait=1, keep=True)
                        show_img(fake_image, 'target', wait=1, keep=True)

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