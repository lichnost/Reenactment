import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import numpy as np
from sklearn.metrics import auc
from utils import *
import torch_optimizer as optim
from torch.optim import lr_scheduler
from torch.nn import init
import math
import os
from types import SimpleNamespace

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

    estimator = Estimator(stacks=arg.hour_stack, msg_pass=arg.msg_pass)

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

    regressor = Regressor(fuse_stages=arg.fuse_stage, output=2*kp_num[resume_dataset])

    if resume_epoch > 0:
        load_path = arg.resume_folder + resume_dataset+'_regressor_' + str(resume_epoch) + '.pth'
        print('Loading regressor from ' + load_path)
        regressor = load_weights(regressor, load_path, devices_list[0])

    if arg.cuda:
        regressor = regressor.cuda(device=devices_list[0])

    return regressor


def create_model_heatmap_discrim(arg, devices_list, eval=False):
    from models import HeatmapDiscrim
    resume_epoch = arg.eval_epoch_boundary_discriminator if eval else arg.resume_epoch

    discrim = HeatmapDiscrim()

    if resume_epoch > 0:
        load_path = arg.resume_folder + 'discrim_boundary_' + str(resume_epoch) + '.pth'
        print('Loading discriminator from ' + load_path)
        discrim = load_weights(discrim, load_path, devices_list[0])

    if arg.cuda:
        discrim = discrim.cuda(device=devices_list[0])

    return discrim


def create_model_decoder_discrim(arg, devices_list, eval=False):
    from models import DecoderTransformerDiscrim
    resume_dataset = arg.eval_dataset_decoder if eval else arg.dataset
    resume_split = arg.eval_split_decoder if eval else arg.split
    resume_epoch = arg.eval_epoch_decoder_discriminator if eval else arg.resume_epoch

    discrim = DecoderTransformerDiscrim()

    if resume_epoch > 0:
        load_path = arg.resume_folder + 'decoder_discrim_'+resume_dataset+'_'+resume_split+'_'+ str(resume_epoch) + '.pth'
        print('Loading decoder discriminator from ' + load_path)
        discrim = load_weights(discrim, load_path, devices_list[0])
    else:
        init_weights(discrim)

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
    else:
        init_weights(decoder)

    if arg.cuda:
        decoder = decoder.cuda(device=devices_list[0])

    return decoder


def create_model_pca(arg, devices_list, eval=False, inverse=False):
    from models import PCA
    resume_dataset = arg.eval_dataset_pca if eval else arg.dataset
    resume_split = arg.eval_split_pca if eval else arg.split
    resume_split_source = arg.eval_split_source_pca if eval else arg.split_source

    pca = PCA(in_size=2*kp_num[arg.dataset], pca_size=arg.pca_components)

    suffix = '_pca_inverse' if inverse else '_pca'

    load_path_first = arg.resume_folder + resume_dataset + '_' + resume_split + '+' + resume_split_source + suffix + '.pth'
    load_path_second = arg.resume_folder + resume_dataset + '_' + resume_split_source + '+' + resume_split + suffix + '.pth'

    if os.path.exists(load_path_first):
        print('Loading PCA from ' + load_path_first)
        pca = load_weights(pca, load_path_first, devices_list[0])

    if os.path.exists(load_path_second):
        print('Loading PCA from ' + load_path_second)
        pca = load_weights(pca, load_path_second, devices_list[0])

    if arg.cuda:
        pca = pca.cuda(device=devices_list[0])

    return pca


def create_model_transformer_a2b(arg, devices_list, eval=False):
    from models import Transformer
    resume_dataset = arg.eval_dataset_transformer if eval else arg.dataset
    resume_a = arg.eval_split_source_transformer if eval else arg.split_source
    resume_b = arg.eval_split_transformer if eval else arg.split
    resume_epoch = arg.eval_epoch_transformer if eval else arg.resume_epoch

    transformer = Transformer(in_channels=boundary_num, out_channels=boundary_num)

    if resume_epoch > 0:
        load_path = arg.resume_folder + 'transformer_'+resume_dataset+'_'+resume_a+'2'+resume_b+'_' + str(resume_epoch) + '.pth'
        print('Loading Transformer from ' + load_path)
        transformer = load_weights(transformer, load_path, devices_list[0])
    else:
        init_weights(transformer, init_type='transformer')
        # init_weights(transformer)

    if arg.cuda:
        transformer = transformer.cuda(device=devices_list[0])

    return transformer


def create_model_transformer_b2a(arg, devices_list, eval=False):
    from models import Transformer
    resume_dataset = arg.eval_dataset_transformer if eval else arg.dataset
    resume_b = arg.eval_split_source_trasformer if eval else arg.split_source
    resume_a = arg.eval_split_trasformer if eval else arg.split
    resume_epoch = arg.eval_epoch_transformer if eval else arg.resume_epoch

    transformer = Transformer(in_channels=boundary_num, out_channels=boundary_num)

    if resume_epoch > 0:
        load_path = arg.resume_folder + 'transformer_'+resume_dataset+'_'+resume_b+'2'+resume_a+'_' + str(resume_epoch) + '.pth'
        print('Loading Transformer from ' + load_path)
        transformer = load_weights(transformer, load_path, devices_list[0])
    else:
        init_weights(transformer, init_type='transformer')
        # init_weights(transformer)

    if arg.cuda:
        transformer = transformer.cuda(device=devices_list[0])

    return transformer


def create_model_transformer_discrim_a(arg, devices_list, eval=False):
    from models import DecoderTransformerDiscrim
    resume_dataset = arg.eval_dataset_pca if eval else arg.dataset
    resume_split = arg.eval_split_source_trasformer if eval else arg.split_source
    resume_epoch = arg.eval_epoch_pca if eval else arg.resume_epoch

    discrim = DecoderTransformerDiscrim(in_channels=boundary_num)

    if resume_epoch > 0:
        load_path = arg.resume_folder + 'transformer_discrim_'+resume_dataset+'_'+resume_split+'_' + str(resume_epoch) + '.pth'
        print('Loading transformer discriminator from ' + load_path)
        discrim = load_weights(discrim, load_path, devices_list[0])
    else:
        init_weights(discrim, init_type='transformer')
        # init_weights(discrim)

    if arg.cuda:
        discrim = discrim.cuda(device=devices_list[0])

    return discrim


def create_model_transformer_discrim_b(arg, devices_list, eval=False):
    from models import DecoderTransformerDiscrim
    resume_dataset = arg.eval_dataset_pca if eval else arg.dataset
    resume_split = arg.eval_split_trasformer if eval else arg.split
    resume_epoch = arg.eval_epoch_pca if eval else arg.resume_epoch

    discrim = DecoderTransformerDiscrim(in_channels=boundary_num)

    if resume_epoch > 0:
        load_path = arg.resume_folder + 'transformer_discrim_'+resume_dataset+'_'+resume_split+'_' + str(resume_epoch) + '.pth'
        print('Loading transformer discriminator from ' + load_path)
        discrim = load_weights(discrim, load_path, devices_list[0])
    else:
        init_weights(discrim, init_type='transformer')
        # init_weights(discrim)

    if arg.cuda:
        discrim = discrim.cuda(device=devices_list[0])

    return discrim


def create_model_align(arg, devices_list, eval=False):
    from models import Align
    resume_dataset = arg.eval_dataset_align if eval else arg.dataset
    resume_epoch = arg.eval_epoch_align if eval else arg.resume_epoch

    align = Align()

    if resume_epoch > 0:
        load_path = arg.resume_folder + 'align_' + resume_dataset + '_' + str(resume_epoch) + '.pth'
        print('Loading align from ' + load_path)
        align = load_weights(align, load_path, devices_list[0])
    else:
        init_weights(align)

    if arg.cuda:
        align = align.cuda(device=devices_list[0])

    return align


def create_model_edge(arg, devices_list, eval=False):
    from models import Edge

    edge = Edge()
    if arg.cuda:
        edge = edge.cuda(device=devices_list[0])

    return edge


def create_model_flame(arg, devices_list, eval=False):
    from flame.FLAME import TexturedFLAME
    resume_dataset = arg.eval_dataset_align if eval else arg.dataset
    resume_split = arg.eval_split_flame if eval else arg.split
    resume_epoch = arg.eval_epoch_align if eval else arg.resume_epoch

    flame_conf = SimpleNamespace()
    flame_conf.flame_model_path = arg.flame_model_path
    flame_conf.use_face_contour = arg.flame_use_face_contour
    flame_conf.batch_size = arg.batch_size
    flame_conf.shape_params = arg.flame_shape_params
    flame_conf.expression_params = arg.flame_expression_params
    flame_conf.pose_params = arg.flame_pose_params
    flame_conf.use_3D_translation = False
    flame_conf.static_landmark_embedding_path = arg.flame_static_landmark_embedding_path
    flame_conf.dynamic_landmark_embedding_path = arg.flame_dynamic_landmark_embedding_path
    flame_conf.texture_path = arg.flame_texture_path

    align = TexturedFLAME(flame_conf, arg.crop_size, devices_list[0])

    if resume_epoch > 0:
        load_path = arg.resume_folder + 'flame_' + resume_dataset + '_' + resume_split + '_' + str(resume_epoch) + '.pth'
        print('Loading FLAME from ' + load_path)
        align = load_weights(align, load_path, devices_list[0])

    if arg.cuda:
        align = align.cuda(device=devices_list[0])

    return align


def create_model_segment(arg, devices_list, eval=False):
    from segment import BiSeNet

    net = BiSeNet(19)

    if arg.segment_model_path is None or not os.path.exists(arg.segment_model_path):
        raise FileNotFoundError()

    load_path = arg.segment_model_path
    print('Loading segmentation model from ' + load_path)
    net = load_weights(net, load_path, devices_list[0])

    if arg.cuda:
        net = net.cuda(device=devices_list[0])

    return net


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

def eye_centers(coords, dataset):
    if left_eye_center_index_x[dataset].__class__ != list:
        left_center = [coords[left_eye_center_index_x[dataset]], coords[left_eye_center_index_y[dataset]]]
        right_center = [coords[right_eye_center_index_x[dataset]], coords[right_eye_center_index_y[dataset]]]

        return left_center, right_center
    else:
        length = len(left_eye_center_index_x[dataset])
        l_eye_x_avg, l_eye_y_avg, r_eye_x_avg, r_eye_y_avg = 0., 0., 0., 0.
        for i in range(length):
            l_eye_x_avg += coords[left_eye_center_index_x[dataset][i]]
            l_eye_y_avg += coords[left_eye_center_index_y[dataset][i]]
            r_eye_x_avg += coords[right_eye_center_index_x[dataset][i]]
            r_eye_y_avg += coords[right_eye_center_index_y[dataset][i]]
        l_eye_x_avg /= length
        l_eye_y_avg /= length
        r_eye_x_avg /= length
        r_eye_y_avg /= length

        left_center = [l_eye_x_avg, l_eye_y_avg]
        right_center = [r_eye_x_avg, r_eye_y_avg]

        return left_center, right_center



def calc_normalize_factor(dataset, gt_coords_xy, normalize_way='inter_pupil'):
    if normalize_way == 'inter_ocular':
        error_normalize_factor = np.sqrt(
            (gt_coords_xy[0][left_eye_left_corner_index_x[dataset]] - gt_coords_xy[0][right_eye_right_corner_index_x[dataset]]) *
            (gt_coords_xy[0][left_eye_left_corner_index_x[dataset]] - gt_coords_xy[0][right_eye_right_corner_index_x[dataset]]) +
            (gt_coords_xy[0][left_eye_left_corner_index_y[dataset]] - gt_coords_xy[0][right_eye_right_corner_index_y[dataset]]) *
            (gt_coords_xy[0][left_eye_left_corner_index_y[dataset]] - gt_coords_xy[0][right_eye_right_corner_index_y[dataset]]))
        return error_normalize_factor
    elif normalize_way == 'inter_pupil':
        left_center, right_center = eye_centers(gt_coords_xy, dataset)
        error_normalize_factor = np.sqrt((left_center[0] - right_center[0]) * (left_center[0] - right_center[0]) +
                                         (left_center[1] - right_center[1]) * (left_center[1] - right_center[1]))
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


def get_heatmap_gray(heatmaps, denorm=False, denorm_base=255, cutoff=False):
    result = torch.sum(heatmaps, dim=heatmaps.ndim-3)
    if cutoff:
        result[result > 1] = 1
    if denorm:
        result = result - torch.min(result)
        result = result * (denorm_base / torch.max(result))
    return result

def coord_transform(xy, crop_matrix):
    return (crop_matrix[0][0] * xy[0] + crop_matrix[0][1] * xy[1] + crop_matrix[0][2],
            crop_matrix[1][0] * xy[0] + crop_matrix[1][1] * xy[1] + crop_matrix[1][2])


def create_optimizer(arg, parameters, create_scheduler=False, discrim=False):
    lr = arg.lr_discrim if discrim else arg.lr
    weight_decay = arg.weight_decay_discrim if discrim else arg.weight_decay
    if arg.optimizer == 'Lamb':
        optimizer = optim.Lamb(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.5, 0.999)
        )
    elif arg.optimizer == 'AdaBound':
        optimizer = optim.AdaBound(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.5, 0.999)
        )
    elif arg.optimizer == 'Yogi':
        optimizer = optim.Yogi(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.5, 0.999)
        )
    elif arg.optimizer == 'DiffGrad':
        optimizer = optim.DiffGrad(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.5, 0.999)
        )
    elif arg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.5, 0.999)
        )
    else:
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=arg.momentum,
            weight_decay=weight_decay
        )

    if create_scheduler:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=4, threshold=1e-2, verbose=True)
    else:
        scheduler = None

    return optimizer, scheduler


def normalized_bbox(coords, dataset, resize_type='height', face_size=0.4, top_shift=0.5):
    points_num = kp_num[dataset]

    if resize_type == 'width':
        coords_x = coords[:2 * points_num:2]
        width = (coords_x.max() - coords_x.min()) / face_size
        height = width
    else:
        coords_y = coords[1:2 * points_num:2]
        height = (coords_y.max() - coords_y.min()) / face_size
        width = height

    left_center, right_center = eye_centers(coords, dataset)
    centers = np.mean(np.array([left_center, right_center]), axis=0)

    return [
        centers[0] - width / 2,
        centers[1] - height * top_shift,
        centers[0] + width / 2,
        centers[1] + height * (1 - top_shift)
    ]

def detect_coords(arg, img, bbox, crop_size, estimator, regressor, devices):
    position_before = np.float32([
        [int(bbox[0]), int(bbox[1])],
        [int(bbox[0]), int(bbox[3])],
        [int(bbox[2]), int(bbox[3])]
    ])
    position_after = np.float32([[0, 0],
                                 [0, crop_size - 1],
                                 [crop_size - 1, crop_size - 1]])
    crop_matrix = cv2.getAffineTransform(position_before, position_after)
    inv_crop_matrix = cv2.invertAffineTransform(crop_matrix)

    img_color = cv2.warpAffine(img, crop_matrix, (crop_size, crop_size))

    # cv2.imshow('img_crop',
    #            img[bbox[1]:bbox[3], bbox[0]:bbox[2]])
    # cv2.waitKey()
    # cv2.destroyWindow('img_crop')
    #
    # cv2.imshow('img_color', img_color)
    # cv2.waitKey()
    # cv2.destroyWindow('img_color')

    img_color = np.float32(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    img = convert_img_to_gray(img_color)
    img = pic_normalize_gray(img)

    input = torch.Tensor(img)
    input = input.unsqueeze(0).unsqueeze(0)
    if arg.cuda:
        input = input.cuda(device=devices[0])

    heatmap = estimator(input)[-1]
    coords = regressor(input, heatmap).detach().cpu().squeeze().numpy()  # output x and y: points_num*2
    return coords, crop_matrix, inv_crop_matrix, heatmap


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def weights_init_transformer(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('DeformConvNet') == -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'transformer':
        net.apply(weights_init_transformer)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)



def face_orientation(size, landmarks, model):
    """
    Face orientation detection.
    :param size: (width, height)
    :param landmarks: array of coordinates as tuples
        [(Nose tip x, Nose tip y),
         (Chin x, Chin y),
         (Left eye left corner x, Left eye left corner y),
         (Right eye right corner x, Right eye right corner y),
         (Left Mouth corner x, Left Mouth corner y),
         (Right mouth corner x, Right mouth corner y)]
    :param model: array of coordinates as tuples
        [(Nose tip x, Nose tip y),
         (Chin x, Chin y),
         (Left eye left corner x, Left eye left corner y),
         (Right eye right corner x, Right eye right corner y),
         (Left Mouth corner x, Left Mouth corner y),
         (Right mouth corner x, Right mouth corner y)]
    :return:
    """

    image_points = np.array(landmarks, dtype="double")
    model = np.array(model, dtype="double")

    generic = np.array([
        [0.0, 0.0, 0.0],  # Nose tip
        [0.0, -330.0, -65.0],  # Chin
        [-170.0, 170.0, -135.0],  # Left eye left corner
        [170.0, 170.0, -135.0],  # Right eye right corner
        [-145.0, -150.0, -125.0],  # Left Mouth corner
        [145.0, -150.0, -125.0]  # Right mouth corner
    ])

    # norm_model = np.linalg.norm(model[5] - model[4])
    # norm_generic = np.linalg.norm(generic[5, :1] - generic[4, :1])
    # model_scale = norm_model / norm_generic
    # generic *= model_scale

    # generic[:, :2] = model[:, :]
    model_points = generic

    # Camera internals

    center = (size[0] / 2, size[1] / 2)
    focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    axis = np.float32([[500, 0, 0],
                       [0, 500, 0],
                       [0, 0, 500]])

    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return np.array(imgpts).reshape([3, 2]), np.array(modelpts).reshape([6, 2]), (str(int(roll)), str(int(pitch)), str(int(yaw))), landmarks[0]


def orientation_landmarks(dataset, shape):
    nose = [shape[nose_tip_x[dataset]], shape[nose_tip_y[dataset]]]
    chin = [shape[chin_bottom_x[dataset]], shape[chin_bottom_y[dataset]]]
    left_eye_left = [shape[left_eye_left_corner_index_x[dataset]], shape[left_eye_left_corner_index_y[dataset]]]
    right_eye_right = [shape[right_eye_right_corner_index_x[dataset]], shape[right_eye_right_corner_index_y[dataset]]]

    if left_mouth_x[dataset].__class__ != list:
        left_mouth = [shape[left_mouth_x[dataset]], shape[left_mouth_y[dataset]]]
        right_mouth = [shape[right_mouth_x[dataset]], shape[right_mouth_y[dataset]]]
    else:
        length = len(left_mouth_x[dataset])
        left_mouth_x_avg, left_mouth_y_avg, right_mouth_x_avg, right_mouth_y_avg = 0., 0., 0., 0.
        for i in range(length):
            left_mouth_x_avg += shape[left_mouth_x[dataset][i]]
            left_mouth_y_avg += shape[left_mouth_y[dataset][i]]
            right_mouth_x_avg += shape[right_mouth_x[dataset][i]]
            right_mouth_y_avg += shape[right_mouth_y[dataset][i]]
        left_mouth_x_avg /= length
        left_mouth_y_avg /= length
        right_mouth_x_avg /= length
        right_mouth_y_avg /= length

        left_mouth = [left_mouth_x_avg, left_mouth_y_avg]
        right_mouth = [right_mouth_x_avg, right_mouth_y_avg]

    return [nose, chin, left_eye_left, right_eye_right, left_mouth, right_mouth]


def generate_random(ranges):
    starts = ranges[:, 0]
    widths = ranges[:, 1] - ranges[:, 0]
    return np.float32(starts + widths * np.random.random(ranges.shape[0]))


def rescale_0_1(data, min, range):
    if isinstance(min, torch.Tensor) and isinstance(range, torch.Tensor)\
            and min.ndim > 0 and min.shape[0] == 3 and range.ndim> 0 and range.shape[0] == 3:
        range = torch.where(range < 1e-6, torch.ones_like(range), range)
        data = data - min[None, :, None, None]
        data = data / range[None, :, None, None]
    else:
        range = 1 if range < 1e-6 else range
        data = data - min
        data = data / range
    return data


def derescale_0_1(data, min, range):
    if isinstance(min, torch.Tensor) and isinstance(range, torch.Tensor)\
            and min.ndim > 0 and min.shape[0] == 3 and range.ndim> 0 and range.shape[0] == 3:
        range = torch.where(range < 1e-6, torch.ones_like(range), range)
        data = data * range[None, :, None, None]
        data = data + min[None, :, None, None]
    else:
        range = 1 if range < 1e-6 else range
        data = data * range
        data = data + min
    return data