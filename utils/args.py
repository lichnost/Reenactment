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
parser.add_argument('--dataset_source',     default='Faces',             type=str)
parser.add_argument('--split',              default='train',             type=str)
parser.add_argument('--split_source',       default='PavelSemenov',      type=str)
parser.add_argument('--type',               default='train',             type=str, choices=['train', 'eval'])


# dataloader
parser.add_argument('--crop_size',          default=256,                 type=int)
parser.add_argument('--batch_size',         default=14,                   type=int) # not gt_heatmaps 14, gt_heatmaps 230
parser.add_argument('--workers',            default=8,                   type=int)
parser.add_argument('--shuffle',            default=True,                type=str2bool)
parser.add_argument('--PDB',                default=False,                type=str2bool) # not gt_heatmaps False, gt_heatmaps True
parser.add_argument('--RGB',                default=False,                type=str2bool)
parser.add_argument('--trans_ratio',        default=0.2,                 type=float) # boundary=0.2
parser.add_argument('--rotate_limit',       default=45.,                 type=float) # boundary=45.
parser.add_argument('--scale_ratio_up',     default=0.1,                 type=float) # boundary=0.2
parser.add_argument('--scale_ratio_down',   default=0.2,                 type=float) # boundary=0.4
parser.add_argument('--scale_vertical',     default=0.2,                 type=float) # boundary=0.2
parser.add_argument('--scale_horizontal',   default=0.2,                 type=float) # boundary=0.2

# devices
parser.add_argument('--cuda',               default=True,                type=str2bool)
parser.add_argument('--gpu_id',             default='0',                 type=str)

# learning parameters
parser.add_argument('--optimizer',          default='Yogi',          type=str,
                    choices=['Lamb', 'SGD', 'Adam', 'Yogi'])
parser.add_argument('--momentum',           default=0.1,                 type=float)
parser.add_argument('--weight_decay',       default=0,                type=float) # decoder=0
parser.add_argument('--lr',                 default=1e-4,                type=float) # decoder=2e-2
parser.add_argument('--gamma',              default=0.2,                 type=float)
parser.add_argument('--step_values',        default=[],        type=list)
parser.add_argument('--max_epoch',          default=1500,                type=int)
parser.add_argument('--lr_scheduler',          default=None,          type=str,
                    choices=['Lamb', 'SGD', 'Adam'])
parser.add_argument('--boundary_cutoff_lambda',          default=0.4,                type=float)

# losses setting
parser.add_argument('--loss_type',          default='L1',          type=str,
                    choices=['L1', 'L2', 'smoothL1', 'wingloss', 'adaptiveWingloss']) # boundary=wingloss, transformer=L1
parser.add_argument('--gp_loss',     default=True,                type=str2bool)
parser.add_argument('--loss_gp_lambda',     default=1.,                 type=float) # decoder=10. transformer=5.
parser.add_argument('--cp_loss',     default=False,                type=str2bool)
parser.add_argument('--loss_cp_lambda',     default=0.01,                 type=float) # decoder=0.01
parser.add_argument('--wingloss_omega',         default=10,                  type=float)
parser.add_argument('--wingloss_epsilon',         default=2,                   type=float)
parser.add_argument('--wingloss_theta',         default=0.5,                  type=float)
parser.add_argument('--wingloss_alpha',         default=2.1,                   type=float)
parser.add_argument('--regressor_loss',     default=False,                type=str2bool)
parser.add_argument('--loss_regressor_lambda',     default=1.0,                type=float)
parser.add_argument('--feature_loss',     default=True,                type=str2bool)
parser.add_argument('--feature_loss_type',          default='relu2_2_and_relu3_3',          type=str,
                    choices=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu1_2_and_relu2_2', 'relu2_2_and_relu3_3', 'relu3_3_and_relu4_3'])
parser.add_argument('--loss_feature_lambda',     default=10.,                type=float)
parser.add_argument('--loss_pixel_lambda',     default=100.,                type=float)
parser.add_argument('--loss_discrim_lambda',     default=5.0,                type=float) # transformer=1.0
# parser.add_argument('--loss_cycle_lambda',     default=1.0,                type=float) # transformer=10.0
parser.add_argument('--loss_pca_lambda',     default=10.,                type=float) # transformer=10.0

# pca
parser.add_argument('--pca_components',          default=20,                type=int)
parser.add_argument('--pca_used_components',     default=3,                type=int)

# resume training parameters
parser.add_argument('--resume_epoch',       default=0,                   type=int)
parser.add_argument('--resume_folder',      default='./weights/checkpoints/',  type=str)

# model saving parameters
parser.add_argument('--save_folder',        default='./weights/',        type=str)
parser.add_argument('--save_interval',      default=10,                 type=int)

# model setting
parser.add_argument('--hour_stack',         default=4,                   type=int)
parser.add_argument('--msg_pass',           default=True,                type=str2bool)
parser.add_argument('--GAN',                default=True,                type=str2bool) # decoder=False
parser.add_argument('--fuse_stage',         default=4,                   type=int)
parser.add_argument('--sigma',              default=1.0,                 type=float)
parser.add_argument('--theta',              default=1.5,                 type=float)
parser.add_argument('--delta',              default=0.8,                 type=float)

# aligner settings
parser.add_argument('--video_path',         default=None,                 type=str)

# evaluate parameters
parser.add_argument('--eval_dataset_regressor',       default='WFLW',              type=str)
parser.add_argument('--eval_dataset_decoder',       default='WFLW',              type=str)
parser.add_argument('--eval_dataset_pca',       default='WFLW',              type=str)
parser.add_argument('--eval_dataset_align',       default='WFLW',              type=str)
parser.add_argument('--eval_split_decoder',         default='PavelSemenov',              type=str)
parser.add_argument('--eval_epoch_estimator',         default=960,                 type=int)
parser.add_argument('--eval_epoch_regressor',         default=960,                 type=int)
parser.add_argument('--eval_epoch_boundary_discriminator',     default=960,                 type=int)
parser.add_argument('--eval_epoch_decoder',          default=85,                 type=int)
parser.add_argument('--eval_epoch_decoder_discriminator',     default=85,                 type=int)
parser.add_argument('--eval_epoch_pca',          default=150000,                 type=int)
parser.add_argument('--eval_epoch_align',          default=350,                 type=int)
parser.add_argument('--max_threshold',      default=0.1,                 type=float)
parser.add_argument('--norm_way',           default='inter_ocular',      type=str,
                    choices=['inter_pupil', 'inter_ocular', 'face_size'])
parser.add_argument('--eval_visual',        default=True ,               type=str2bool)
parser.add_argument('--save_img',           default=False,                type=str2bool)
parser.add_argument('--realtime',           default=True,              type=str2bool)
parser.add_argument('--normalized_bbox',    default=True,              type=str2bool)
parser.add_argument('--eval_video_path',       default=None,              type=str)

def parse_args():
    result = parser.parse_args()

    if len(result.step_values) > 0:
        assert result.resume_epoch < result.step_values[0]
        assert result.resume_epoch < result.max_epoch
        assert result.step_values[-1] < result.max_epoch

    result.dataset_route = dataset_info.init_dataset_route(result.dataset_route)

    return result
