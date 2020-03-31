from .dataset_info import *

import cv2
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline as spline


def show_img(pic, name='pic', x=-1, y=-1, wait=0, keep=False):
    cv2.imshow(name, pic)
    if x != -1 and y != -1:
        cv2.moveWindow(name, x, y)
    if keep is False:
        cv2.waitKey(wait)
        cv2.destroyAllWindows()


def watch_gray_heatmap(gt_heatmap):
    heatmap_sum = gt_heatmap[0]
    for index in range(boundary_num - 1):
        heatmap_sum += gt_heatmap[index + 1]
    show_img(heatmap_sum, 'heatmap_sum')


def watch_pic_kp(dataset, pic, kp):
    for kp_index in range(kp_num[dataset]):
        cv2.circle(
            pic,
            (int(kp[2*kp_index]), int(kp[2*kp_index+1])),
            2,
            (0, 0, 255),
            -1
        )
    show_img(pic)


def watch_pic_kp_xy(dataset, pic, coord_x, coord_y):
    for kp_index in range(kp_num[dataset]):
        cv2.circle(
            pic,
            (int(coord_x[kp_index]), int(coord_y[kp_index])),
            1,
            (0, 0, 255)
        )
    show_img(pic)


def eval_heatmap(arg, heatmaps, img_name, bbox, save_img=False):
    heatmaps = F.interpolate(heatmaps, scale_factor=4, mode='bilinear', align_corners=True)
    heatmaps = heatmaps.squeeze(0).detach().cpu().numpy()
    heatmaps_sum = heatmaps[0]
    for heatmaps_index in range(boundary_num-1):
        heatmaps_sum += heatmaps[heatmaps_index+1]
    plt.axis('off')
    plt.imshow(heatmaps_sum, interpolation='nearest', vmax=1., vmin=0.)
    if save_img:
        import os
        if not os.path.exists('./imgs'):
            os.mkdir('./imgs')
        fig = plt.gcf()
        fig.set_size_inches(2.56 / 3, 2.56 / 3)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        name = (img_name[0]).split('/')[-1]
        fig.savefig('./imgs/'+name.split('.')[0]+'_hm.png', format='png', transparent=True, dpi=300, pad_inches=0)

        pic = cv2.imread(arg.dataset_route[arg.dataset] + img_name[0])
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
        pic = cv2.warpAffine(pic, crop_matrix, (arg.crop_size, arg.crop_size))
        cv2.imwrite('./imgs/' + name.split('.')[0] + '_init.png', pic)
        hm = cv2.imread('./imgs/'+name.split('.')[0]+'_hm.png')
        syn = cv2.addWeighted(pic, 0.4, hm, 0.6, 0)
        cv2.imwrite('./imgs/'+name.split('.')[0]+'_syn.png', syn)
    else:
        plt.show()


def eval_pred_points(arg, pred_coords, img_name, bbox, save_img=False):
    pic = cv2.imread(arg.dataset_route[arg.dataset] + img_name[0])
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
    pic = cv2.warpAffine(pic, crop_matrix, (arg.crop_size, arg.crop_size))

    for coord_index in range(kp_num[arg.dataset]):
        cv2.circle(pic, (int(pred_coords[2 * coord_index]), int(pred_coords[2 * coord_index + 1])), 2, (0, 0, 255))
    if save_img:
        import os
        if not os.path.exists('./imgs'):
            os.mkdir('./imgs')
        name = (img_name[0]).split('/')[-1]
        cv2.imwrite('./imgs/'+name.split('.')[0]+'_lmk.png', pic)
    else:
        show_img(pic)


def eval_gt_pred_points(arg, gt_coords, pred_coords, img_name, bbox, save_img=False):
    pic = cv2.imread(arg.dataset_route[arg.dataset] + img_name[0])
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
    pic = cv2.warpAffine(pic, crop_matrix, (arg.crop_size, arg.crop_size))

    for coord_index in range(kp_num[arg.dataset]):
        cv2.circle(pic, (int(pred_coords[2 * coord_index]), int(pred_coords[2 * coord_index + 1])), 2, (0, 0, 255))
        cv2.circle(pic, (int(gt_coords[0, 2 * coord_index]), int(gt_coords[0, 2 * coord_index + 1])), 2, (0, 255, 0))
    if save_img:
        import os
        if not os.path.exists('./imgs'):
            os.mkdir('./imgs')
        name = (img_name[0]).split('/')[-1]
        cv2.imwrite('./imgs/'+name.split('.')[0]+'_lmk.png', pic)
    else:
        show_img(pic)


def eval_CED(auc_record):
    error = np.linspace(0., 0.1, 21)
    error_new = np.linspace(error.min(), error.max(), 300)
    auc_value = np.array([auc_record[0], auc_record[99], auc_record[199], auc_record[299],
                          auc_record[399], auc_record[499], auc_record[599], auc_record[699],
                          auc_record[799], auc_record[899], auc_record[999], auc_record[1099],
                          auc_record[1199], auc_record[1299], auc_record[1399], auc_record[1499],
                          auc_record[1599], auc_record[1699], auc_record[1799], auc_record[1899],
                          auc_record[1999]])
    CFSS_auc_value = np.array([0., 0., 0., 0., 0.,
                               0., 0.02, 0.09, 0.18, 0.30,
                               0.45, 0.60, 0.70, 0.75, 0.79,
                               0.82, 0.85, 0.87, 0.88, 0.89, 0.90])
    SAPM_auc_value = np.array([0., 0., 0., 0., 0.,
                               0., 0., 0., 0.02, 0.08,
                               0.17, 0.28, 0.43, 0.58, 0.71,
                               0.78, 0.83, 0.86, 0.89, 0.91, 0.92])
    TCDCN_auc_value = np.array([0., 0., 0., 0., 0.,
                                0., 0., 0.02, 0.05, 0.10,
                                0.19, 0.29, 0.38, 0.47, 0.56,
                                0.64, 0.70, 0.75, 0.79, 0.82, 0.826])
    auc_smooth = spline(error, auc_value, error_new)
    CFSS_auc_smooth = spline(error, CFSS_auc_value, error_new)
    SAPM_auc_smooth = spline(error, SAPM_auc_value, error_new)
    TCDCN_auc_smooth = spline(error, TCDCN_auc_value, error_new)
    plt.plot(error_new, auc_smooth, 'r-')
    plt.plot(error_new, CFSS_auc_smooth, 'g-')
    plt.plot(error_new, SAPM_auc_smooth, 'y-')
    plt.plot(error_new, TCDCN_auc_smooth, 'm-')
    plt.legend(['LAB, Error: 5.35%, Failure: 4.73%',
                'CFSS, Error: 6.28%, Failure: 9.07%',
                'SAPM, Error: 6.64%, Failure: 5.72%',
                'TCDCN, Error: 7.66%, Failure: 16.17%'], loc=4)
    plt.plot(error, auc_value, 'rs')
    plt.plot(error, CFSS_auc_value, 'go')
    plt.plot(error, SAPM_auc_value, 'y^')
    plt.plot(error, TCDCN_auc_value, 'mx')
    plt.axis([0., 0.1, 0., 1.])
    plt.show()


def draw_circle(img, xy, color=(0, 0, 255)):
    return cv2.circle(img, xy, 2, color, -1)


def draw_text(img, text, xy, color=(0, 0, 255), scale=1):
    return cv2.putText(img, text, xy, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, color=color)


def draw_orientation(img, orientation):
    nose = (int(orientation[3][0]), int(orientation[3][1]))
    points = orientation[0].astype(int)
    cv2.line(img, nose, tuple(points[1].ravel().astype(int)), (0,255,0), 3) #GREEN
    cv2.line(img, nose, tuple(points[0].ravel().astype(int)), (255,0,), 3) #BLUE
    cv2.line(img, nose, tuple(points[2].ravel().astype(int)), (0,0,255), 3) #RED