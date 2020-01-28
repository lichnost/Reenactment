import argparse
from utils import dataset_info

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='LAB')

parser.add_argument('--gt_heatmaps',        default=False,             type=str2bool)
parser.add_argument('--save_logs',          default=False,             type=str2bool)

# dlib
parser.add_argument('--dlib_face_detector_path', default='/home/lichnost/programming/work/ml/models/dlib/mmod_human_face_detector.dat',             type=str)
parser.add_argument('--dlib_shape_predictor_path', default='/home/lichnost/programming/work/ml/models/dlib/shape_predictor_5_face_landmarks.dat',             type=str)

# dataset
parser.add_argument('--dataset_route',      default='/home/lichnost/programming/work/ml/head/data', type=str)
parser.add_argument('--dataset',            default='WFLW',              type=str)
parser.add_argument('--split',              default='train',             type=str)
parser.add_argument('--type',               default='train',             type=str, choices=['train', 'eval'])


# dataloader
parser.add_argument('--crop_size',          default=256,                 type=int)
parser.add_argument('--batch_size',         default=14,                   type=int) # not gt_heatmaps 14, gt_heatmaps 230
parser.add_argument('--workers',            default=2,                   type=int)
parser.add_argument('--shuffle',            default=True,                type=str2bool)
parser.add_argument('--PDB',                default=True,                type=str2bool) # not gt_heatmaps False, gt_heatmaps True
parser.add_argument('--RGB',                default=False,                type=str2bool)
parser.add_argument('--trans_ratio',        default=0.4,                 type=float)
parser.add_argument('--rotate_limit',       default=45.,                 type=float)
parser.add_argument('--scale_ratio',        default=0.4,                 type=float)
parser.add_argument('--scale_vertical',     default=0.0,                 type=float)
parser.add_argument('--scale_horizontal',   default=0.8,                 type=float)

# devices
parser.add_argument('--cuda',               default=True,                type=str2bool)
parser.add_argument('--gpu_id',             default='0',                 type=str)

# learning parameters
parser.add_argument('--momentum',           default=0.9,                 type=float)
parser.add_argument('--weight_decay',       default=5e-5,                type=float)
parser.add_argument('--lr',                 default=8e-5,                type=float)
parser.add_argument('--gamma',              default=0.2,                 type=float)
parser.add_argument('--step_values',        default=[650, 1050],        type=list)
parser.add_argument('--max_epoch',          default=1500,                type=int)

# losses setting
parser.add_argument('--loss_type',          default='adaptiveWingloss',          type=str,
                    choices=['L1', 'L2', 'smoothL1', 'wingloss', 'adaptiveWingloss'])
parser.add_argument('--gp_loss_type',       default='GPFullLoss',            type=str,
                    choices=['GPLoss', 'GPFullLoss', 'HeatmapLoss'])
parser.add_argument('--gp_loss_lambda',     default=100,                 type=float)
parser.add_argument('--cp_loss_lambda',     default=100,                 type=float)
parser.add_argument('--wingloss_omega',         default=10,                  type=int)
parser.add_argument('--wingloss_epsilon',         default=2,                   type=int)
parser.add_argument('--wingloss_theta',         default=0.5,                  type=int)
parser.add_argument('--wingloss_alpha',         default=2.1,                   type=int)
parser.add_argument('--regressor_loss',     default=False,                type=str2bool)
parser.add_argument('--regressor_loss_lambda',     default=1,                type=float)

# resume training parameters
parser.add_argument('--resume_epoch',       default=0,                   type=int)
parser.add_argument('--resume_folder',      default='./weights/checkpoints/',  type=str)

# model saving parameters
parser.add_argument('--save_folder',        default='./weights/',        type=str)
parser.add_argument('--save_interval',      default=5,                 type=int)

# model setting
parser.add_argument('--hour_stack',         default=4,                   type=int)
parser.add_argument('--msg_pass',           default=True,                type=str2bool)
parser.add_argument('--GAN',                default=True,                type=str2bool)
parser.add_argument('--fuse_stage',         default=4,                   type=int)
parser.add_argument('--sigma',              default=1.0,                 type=float)
parser.add_argument('--theta',              default=1.5,                 type=float)
parser.add_argument('--delta',              default=0.8,                 type=float)

# aligner settings
parser.add_argument('--video_path',         default=None,                 type=str)

# evaluate parameters
parser.add_argument('--eval_dataset_regressor',       default='WFLW',              type=str)
parser.add_argument('--eval_dataset_decoder',       default='WFLW',              type=str)
parser.add_argument('--eval_split_decoder',         default='PavelSemenov',              type=str)
parser.add_argument('--eval_epoch_estimator',         default=960,                 type=int)
parser.add_argument('--eval_epoch_regressor',         default=960,                 type=int)
parser.add_argument('--eval_epoch_discriminator',     default=960,                 type=int)
parser.add_argument('--eval_epoch_decoder',          default=85,                 type=int)
parser.add_argument('--max_threshold',      default=0.1,                 type=float)
parser.add_argument('--norm_way',           default='inter_ocular',      type=str,
                    choices=['inter_pupil', 'inter_ocular', 'face_size'])
parser.add_argument('--eval_visual',        default=False ,               type=str2bool)
parser.add_argument('--save_img',           default=False,                type=str2bool)

def parse_args():
    result = parser.parse_args()

    assert result.resume_epoch < result.step_values[0]
    assert result.resume_epoch < result.max_epoch
    assert result.step_values[-1] < result.max_epoch

    result.dataset_route = dataset_info.init_dataset_route(result.dataset_route)

    return result
