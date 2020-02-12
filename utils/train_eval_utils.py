import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import numpy as np
from sklearn.metrics import auc
from utils import *
import torch_optimizer as optim

def get_devices_list(arg):
    devices_list = [torch.device('cpu')]
    if arg.cuda and torch.cuda.is_available():
        devices_list = []
        for dev in arg.gpu_id.split(','):
            devices_list.append(torch.device('cuda:'+dev))
        cudnn.benchmark = True
        cudnn.enabled = True
    return devices_list


def load_weights(net, pth_file, device):
    state_dict = torch.load(pth_file, map_location=device)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    return net


def create_model_estimator(arg, devices_list, eval=False):
    from models import Estimator
    resume_epoch = arg.eval_epoch_estimator if eval else arg.resume_epoch

    estimator = Estimator(gp_loss_lambda=arg.gp_loss_lambda,
                          stacks=arg.hour_stack, msg_pass=arg.msg_pass)

    if resume_epoch > 0:
        load_path = arg.resume_folder + 'estimator_' + str(resume_epoch) + '.pth'
        print('Loading estimator from ' + load_path)
        estimator = load_weights(estimator, load_path, devices_list[0])

    if arg.cuda:
        estimator = estimator.cuda(device=devices_list[0])

    return estimator


def create_model_regressor(arg, devices_list, eval=False):
    from models import Regressor
    resume_dataset = arg.eval_dataset_regressor if eval else arg.dataset
    resume_epoch = arg.eval_epoch_regressor if eval else arg.resume_epoch

    regressor = Regressor(fuse_stages=arg.fuse_stage, output=2*kp_num[arg.dataset])

    if resume_epoch > 0:
        load_path = arg.resume_folder + resume_dataset+'_regressor_' + str(resume_epoch) + '.pth'
        print('Loading regressor from ' + load_path)
        regressor = load_weights(regressor, load_path, devices_list[0])

    if arg.cuda:
        regressor = regressor.cuda(device=devices_list[0])

    return regressor


def create_model_heatmap_discrim(arg, devices_list, eval=False):
    from models import HeatmapDiscrim
    resume_epoch = arg.eval_epoch_discriminator if eval else arg.resume_epoch

    discrim = HeatmapDiscrim()

    if resume_epoch > 0:
        load_path = arg.resume_folder + 'discrim_' + str(resume_epoch) + '.pth'
        print('Loading discriminator from ' + load_path)
        discrim = load_weights(discrim, load_path, devices_list[0])

    if arg.cuda:
        discrim = discrim.cuda(device=devices_list[0])

    return discrim


def create_model_decoder(arg, devices_list, eval=False):
    from models import Decoder
    resume_dataset = arg.eval_dataset_decoder if eval else arg.dataset
    resume_split = arg.eval_split_decoder if eval else arg.split
    resume_epoch = arg.eval_epoch_decoder if eval else arg.resume_epoch

    decoder = Decoder()

    if resume_epoch > 0:
        load_path = arg.resume_folder + 'decoder_' + resume_dataset + '_' + resume_split + '_' + str(resume_epoch) + '.pth'
        print('Loading decoder from ' + load_path)
        decoder = load_weights(decoder, load_path, devices_list[0])

    if arg.cuda:
        decoder = decoder.cuda(device=devices_list[0])

    return decoder


def create_model_pca(arg, devices_list, eval=False):
    from models import PCA
    resume_dataset = arg.eval_dataset_pca if eval else arg.dataset
    resume_epoch = arg.eval_epoch_pca if eval else arg.resume_epoch

    pca = PCA(in_size=2*kp_num[arg.dataset], pca_size=arg.pca_components)

    if resume_epoch > 0:
        load_path = arg.resume_folder + resume_dataset+'_pca_' + str(resume_epoch) + '.pth'
        print('Loading PCA from ' + load_path)
        pca = load_weights(pca, load_path, devices_list[0])

    if arg.cuda:
        pca = pca.cuda(device=devices_list[0])

    return pca


def calc_d_fake(dataset, pred_coords, gt_coords, bcsize, bcsize_set, delta, theta):
    error_regressor = (pred_coords - gt_coords) ** 2
    dist_regressor = torch.zeros(bcsize, kp_num[dataset])
    dfake = torch.zeros(bcsize_set, boundary_num)
    for batch in range(bcsize):
        dist_regressor[batch, :] = \
            (error_regressor[batch][:2*kp_num[dataset]:2] + error_regressor[batch][1:2*kp_num[dataset]:2]) \
            < theta*theta
    for batch_index in range(bcsize):
        for boundary_index in range(boundary_num):
            for kp_index in range(
                    point_range[dataset][boundary_index][0],
                    point_range[dataset][boundary_index][1]
            ):
                if dist_regressor[batch_index][kp_index] == 1:
                    dfake[batch_index][boundary_index] += 1
            if boundary_keys[boundary_index] in boundary_special.keys() and \
                    dataset in boundary_special[boundary_keys[boundary_index]] and \
                    dist_regressor[batch_index][duplicate_point[dataset][boundary_keys[boundary_index]]] == 1:
                dfake[batch_index][boundary_index] += 1
        for boundary_index in range(boundary_num):
            if dfake[batch_index][boundary_index] / point_num_per_boundary[dataset][boundary_index] < delta:
                dfake[batch_index][boundary_index] = 0.
            else:
                dfake[batch_index][boundary_index] = 1.
    if bcsize < bcsize_set:
        for batch_index in range(bcsize, bcsize_set):
            dfake[batch_index] = dfake[batch_index - bcsize]
    return dfake


def calc_normalize_factor(dataset, gt_coords_xy, normalize_way='inter_pupil'):
    if normalize_way == 'inter_ocular':
        error_normalize_factor = np.sqrt(
            (gt_coords_xy[0][lo_eye_corner_index_x[dataset]] - gt_coords_xy[0][ro_eye_corner_index_x[dataset]]) *
            (gt_coords_xy[0][lo_eye_corner_index_x[dataset]] - gt_coords_xy[0][ro_eye_corner_index_x[dataset]]) +
            (gt_coords_xy[0][lo_eye_corner_index_y[dataset]] - gt_coords_xy[0][ro_eye_corner_index_y[dataset]]) *
            (gt_coords_xy[0][lo_eye_corner_index_y[dataset]] - gt_coords_xy[0][ro_eye_corner_index_y[dataset]]))
        return error_normalize_factor
    elif normalize_way == 'inter_pupil':
        if l_eye_center_index_x[dataset].__class__ != list:
            error_normalize_factor = np.sqrt(
                (gt_coords_xy[0][l_eye_center_index_x[dataset]] - gt_coords_xy[0][r_eye_center_index_x[dataset]]) *
                (gt_coords_xy[0][l_eye_center_index_x[dataset]] - gt_coords_xy[0][r_eye_center_index_x[dataset]]) +
                (gt_coords_xy[0][l_eye_center_index_y[dataset]] - gt_coords_xy[0][r_eye_center_index_y[dataset]]) *
                (gt_coords_xy[0][l_eye_center_index_y[dataset]] - gt_coords_xy[0][r_eye_center_index_y[dataset]]))
            return error_normalize_factor
        else:
            length = len(l_eye_center_index_x[dataset])
            l_eye_x_avg, l_eye_y_avg, r_eye_x_avg, r_eye_y_avg = 0., 0., 0., 0.
            for i in range(length):
                l_eye_x_avg += gt_coords_xy[0][l_eye_center_index_x[dataset][i]]
                l_eye_y_avg += gt_coords_xy[0][l_eye_center_index_y[dataset][i]]
                r_eye_x_avg += gt_coords_xy[0][r_eye_center_index_x[dataset][i]]
                r_eye_y_avg += gt_coords_xy[0][r_eye_center_index_y[dataset][i]]
            l_eye_x_avg /= length
            l_eye_y_avg /= length
            r_eye_x_avg /= length
            r_eye_y_avg /= length
            error_normalize_factor = np.sqrt((l_eye_x_avg - r_eye_x_avg) * (l_eye_x_avg - r_eye_x_avg) +
                                             (l_eye_y_avg - r_eye_y_avg) * (l_eye_y_avg - r_eye_y_avg))
            return error_normalize_factor


def inverse_affine(arg, pred_coords, bbox):
    import copy
    pred_coords = copy.deepcopy(pred_coords)
    for i in range(kp_num[arg.dataset]):
        pred_coords[2 * i] = bbox[0] + pred_coords[2 * i]/(arg.crop_size-1)*(bbox[2] - bbox[0])
        pred_coords[2 * i + 1] = bbox[1] + pred_coords[2 * i + 1]/(arg.crop_size-1)*(bbox[3] - bbox[1])
    return pred_coords


def calc_error_rate_i(dataset, pred_coords, gt_coords_xy, error_normalize_factor):
    temp, error = (pred_coords - gt_coords_xy)**2, 0.
    for i in range(kp_num[dataset]):
        error += np.sqrt(temp[2*i] + temp[2*i+1])
    return error/kp_num[dataset]/error_normalize_factor


def calc_error_rate_i_nparts(dataset, pred_coords, gt_coords_xy, error_normalize_factor):
    assert dataset in nparts.keys()
    temp, error = (pred_coords - gt_coords_xy)**2, [0., 0., 0., 0., 0.]
    for i in range(len(nparts[dataset])):
        for j in range(nparts[dataset][i][0], nparts[dataset][i][1]):
            error[i] += np.sqrt(temp[2*j] + temp[2*j+1])
        error[i] = error[i]/(nparts[dataset][i][1] - nparts[dataset][i][0])/error_normalize_factor
    return error


def calc_auc(dataset, split, error_rate, max_threshold):
    error_rate = np.array(error_rate)
    threshold = np.linspace(0, max_threshold, num=2000)
    accuracys = np.zeros(threshold.shape)
    for i in range(threshold.size):
        accuracys[i] = np.sum(error_rate < threshold[i]) * 1.0 / dataset_size[dataset][split]
    return auc(threshold, accuracys) / max_threshold, accuracys


def get_heatmap_gray(heatmaps):
    result = torch.sum(heatmaps, dim=1)
    return result

def coord_transform(xy, crop_matrix):
    return (crop_matrix[0][0] * xy[0] + crop_matrix[0][1] * xy[1] + crop_matrix[0][2],
            crop_matrix[1][0] * xy[0] + crop_matrix[1][1] * xy[1] + crop_matrix[1][2])


def create_optimizer(arg, parameters):
    if arg.optimizer == 'Lamb':
        optimizer = optim.Lamb(
            parameters,
            lr=arg.lr,
            weight_decay=arg.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            decoder.parameters(),
            lr=arg.lr,
            momentum=arg.momentum,
            weight_decay=arg.weight_decay
        )

    return optimizer