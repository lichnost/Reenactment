from jsonargparse import ArgumentParser, ActionConfigFile
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


parser = ArgumentParser(description='LAB')
parser.add_argument('--cfg', action=ActionConfigFile)

parser.add_argument('--regressor_only',        default=False,             type=str2bool)
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
parser.add_argument('--batch_size',         default=14,                   type=int) # not gt_heatmaps 14, gt_heatmaps 230,
parser.add_argument('--workers',            default=8,                   type=int)
parser.add_argument('--shuffle',            default=True,                type=str2bool)
parser.add_argument('--PDB',                default=False,                type=str2bool) # not gt_heatmaps False, gt_heatmaps True, transformer_preliminary_no_GAN=60, transformer_fine__GAN=60
parser.add_argument('--RGB',                default=False,                type=str2bool)
parser.add_argument('--trans_ratio',        default=0.0,                 type=float) # boundary=0.2, decoder=0.0, transformer=0.0
parser.add_argument('--rotate_limit',       default=0.0,                 type=float) # boundary=45., decoder=0.0, transformer=0.0
parser.add_argument('--scale_ratio_up',     default=0.1,                 type=float) # boundary=0.2, decoder=0.1, transformer=0.1
parser.add_argument('--scale_ratio_down',   default=0.1,                 type=float) # boundary=0.4, decoder=0.1, transformer=0.1
parser.add_argument('--scale_vertical',     default=0.0,                 type=float) # boundary=0.2, decoder=0.0, transformer=0.0
parser.add_argument('--scale_horizontal',   default=0.0,                 type=float) # boundary=0.2, decoder=0.0, transformer=0.0
parser.add_argument('--dataset_indexes',    default=[],                   nargs='*', type=int)

# devices
parser.add_argument('--cuda',               default=True,                type=str2bool)
parser.add_argument('--gpu_id',             default='0',                 type=str)

# learning parameters
parser.add_argument('--optimizer',          default='Adam',          type=str,
                    choices=['Lamb', 'SGD', 'Adam', 'Yogi', 'AdaBound', 'DiffGrad'])
parser.add_argument('--momentum',           default=0.1,                 type=float)
parser.add_argument('--weight_decay',       default=0,                type=float) # decoder=0
parser.add_argument('--weight_decay_discrim',       default=0.9,                type=float) # decoder=0, transformer_fine=0.9
parser.add_argument('--lr',                 default=5e-5,                type=float) # decoder=2e-2, transformer_preliminary=5e-4, transformer_fine=5e-5
parser.add_argument('--lr_discrim',                 default=5e-6,                type=float) # transformer_fine=5e-6
parser.add_argument('--gamma',              default=0.2,                 type=float)
parser.add_argument('--step_values',        default=[],        type=list)
parser.add_argument('--max_epoch',          default=1500,                type=int)
parser.add_argument('--lr_scheduler',          default=None,          type=str,
                    choices=['Lamb', 'SGD', 'Adam'])
parser.add_argument('--boundary_cutoff_lambda',          default=0.4,                type=float)

# losses setting
parser.add_argument('--loss_type',          default='L1',          type=str,
                    choices=['L1', 'L2', 'smoothL1', 'wingloss']) # boundary=wingloss, transformer=L1
parser.add_argument('--gp_loss',     default=True,                type=str2bool)
parser.add_argument('--loss_gp_lambda',     default=5.0,                 type=float) # decoder=10. transformer_preliminary=10., transformer_fine=5.0
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
parser.add_argument('--loss_pca_lambda',     default=100.,                type=float) # transformer_preliminary=0.1, transformer_fine=100.
parser.add_argument('--loss_edge_lambda',     default=1000.,                type=float) # transformer_preliminary=0.1, transformer_fine=100.

# pca
parser.add_argument('--pca_components',          default=196,                type=int)
parser.add_argument('--pca_used_components',     default=30,                type=int)
parser.add_argument('--pca_inverse',     default=False,                type=str2bool)

# resume training parameters
parser.add_argument('--resume_epoch',       default=0,                   type=int)
parser.add_argument('--resume_folder',      default='./weights/checkpoints/',  type=str)

# model saving parameters
parser.add_argument('--save_folder',        default='./weights/',        type=str)
parser.add_argument('--save_interval',      default=5,                 type=int)

# model setting
parser.add_argument('--hour_stack',         default=4,                   type=int)
parser.add_argument('--msg_pass',           default=True,                type=str2bool)
parser.add_argument('--GAN',                default=False,                type=str2bool) # decoder=False, transformer_preliminary=False, transformer_fine=False
parser.add_argument('--fuse_stage',         default=4,                   type=int)
parser.add_argument('--sigma',              default=1.0,                 type=float)
parser.add_argument('--theta',              default=1.5,                 type=float)
parser.add_argument('--delta',              default=0.8,                 type=float)

# aligner settings
parser.add_argument('--video_path',         default=None,                 type=str)

# evaluate parameters
parser.add_argument('--eval_dataset_regressor',       default='WFLW',              type=str)
parser.add_argument('--eval_dataset_decoder',       default='Faces',              type=str)
parser.add_argument('--eval_dataset_transformer',       default='Faces',              type=str)
parser.add_argument('--eval_dataset_pca',       default='Faces',              type=str)
parser.add_argument('--eval_split_pca',       default='Adush',              type=str)
parser.add_argument('--eval_split_source_pca',       default='PavelSemenov',              type=str)
parser.add_argument('--eval_dataset_align',       default='WFLW',              type=str)
parser.add_argument('--eval_split_decoder',         default='Adush',              type=str)
parser.add_argument('--eval_split_source_transformer',         default='PavelSemenov',              type=str)
parser.add_argument('--eval_split_transformer',         default='Adush',              type=str)
parser.add_argument('--eval_epoch_estimator',         default=1123,                 type=int)
parser.add_argument('--eval_epoch_regressor',         default=1123,                 type=int)
parser.add_argument('--eval_epoch_boundary_discriminator',     default=960,                 type=int)
parser.add_argument('--eval_epoch_decoder',          default=85,                 type=int)
parser.add_argument('--eval_epoch_decoder_discriminator',     default=85,                 type=int)
parser.add_argument('--eval_epoch_align',          default=350,                 type=int)
parser.add_argument('--eval_epoch_transformer',          default=40,                 type=int)
parser.add_argument('--eval_epoch_source_transformer',          default=40,                 type=int)
parser.add_argument('--max_threshold',      default=0.1,                 type=float)
parser.add_argument('--norm_way',           default='inter_ocular',      type=str,
                    choices=['inter_pupil', 'inter_ocular', 'face_size'])
parser.add_argument('--eval_visual',        default=True ,               type=str2bool)
parser.add_argument('--save_img',           default=True,                type=str2bool)
parser.add_argument('--realtime',           default=True,              type=str2bool)
parser.add_argument('--normalized_bbox',    default=True,              type=str2bool)
parser.add_argument('--normalize_face_size', default=0.4,              type=float)
parser.add_argument('--normalize_top_shift', default=0.5,              type=float)
parser.add_argument('--eval_video_path',       default=None,              type=str)
parser.add_argument('--save_heatmaps',       default=True,                type=str2bool)

# FLAME
parser.add_argument('--flame_model_path',         default='/home/lichnost/programming/work/ml/head/FLAME/model/female_model.pkl',              type=str)
parser.add_argument('--flame_use_face_contour', default=False,               type=str2bool)
parser.add_argument('--flame_shape_params',       default=100,              type=int)
parser.add_argument('--flame_expression_params',  default=50,               type=int)
parser.add_argument('--flame_pose_params',        default=7,               type=int)
parser.add_argument('--flame_use_3D_translation', default=False,               type=str2bool)
parser.add_argument('--flame_static_landmark_embedding_path',       default='/home/lichnost/programming/work/ml/head/FLAME/model/flame_static_embedding.pkl',               type=str)
parser.add_argument('--flame_dynamic_landmark_embedding_path',       default='/home/lichnost/programming/work/ml/head/FLAME/model/flame_dynamic_embedding.npy',             type=str)
parser.add_argument('--flame_texture_path',       default='/home/lichnost/programming/work/ml/head/FLAME/model/FLAME_texture.npz',             type=str)
parser.add_argument('--flame_dataset_mean_shape', default=True,             type=str2bool)
parser.add_argument('--flame_random_params',      default=True,             type=str2bool)
parser.add_argument('--flame_shape_params_path_a',     default=None,             type=str)
parser.add_argument('--flame_shape_params_path_b',     default=None,             type=str)

parser.add_argument('--segment_model_path',       default='/home/lichnost/programming/work/ml/head/Look_At_Boundary_PyTorch/pretrained/face-parsing.PyTorch.pth',             type=str)


def parse_args():
    result = parser.parse_args()

    if len(result.step_values) > 0:
        assert result.resume_epoch < result.step_values[0]
        assert result.resume_epoch < result.max_epoch
        assert result.step_values[-1] < result.max_epoch

    result.dataset_route = dataset_info.init_dataset_route(result.dataset_route)

    return result

