import argparse

parser = argparse.ArgumentParser(description='LAB')

parser.add_argument('--gt_heatmaps',        default=False,             type=bool)
parser.add_argument('--save_logs',          default=False,             type=bool)

# dataset
parser.add_argument('--dataset_route',      default='/home/lichnost/programming/work/ml/head/data', type=str)
parser.add_argument('--dataset',            default='WFLW',              type=str)
parser.add_argument('--split',              default='test',             type=str)

# dataloader
parser.add_argument('--crop_size',          default=256,                 type=int)
parser.add_argument('--batch_size',         default=14,                   type=int) # not gt_heatmaps 14, gt_heatmaps 230
parser.add_argument('--workers',            default=8,                   type=int)
parser.add_argument('--shuffle',            default=True,                type=bool)
parser.add_argument('--PDB',                default=False,                type=bool) # not gt_heatmaps False, gt_heatmaps True
parser.add_argument('--RGB',                default=False,                type=bool)
parser.add_argument('--trans_ratio',        default=0.4,                 type=float)
parser.add_argument('--rotate_limit',       default=45.,                 type=float)
parser.add_argument('--scale_ratio',        default=0.4,                 type=float)
parser.add_argument('--scale_vertical',     default=0.0,                 type=float)
parser.add_argument('--scale_horizontal',   default=0.8,                 type=float)

# devices
parser.add_argument('--cuda',               default=True,                type=bool)
parser.add_argument('--gpu_id',             default='0',                 type=str)

# learning parameters
parser.add_argument('--momentum',           default=0.9,                 type=float)
parser.add_argument('--weight_decay',       default=5e-5,                type=float)
parser.add_argument('--lr',                 default=2e-5,                type=float)
parser.add_argument('--gamma',              default=0.2,                 type=float)
parser.add_argument('--step_values',        default=[1000, 1500],        type=list)
parser.add_argument('--max_epoch',          default=2000,                type=int)

# losses setting
parser.add_argument('--loss_type',          default='wingloss',          type=str,
                    choices=['L1', 'L2', 'smoothL1', 'wingloss'])
parser.add_argument('--gp_loss_type',       default='GPFullLoss',            type=str,
                    choices=['GPLoss', 'GPFullLoss', 'HeatmapLoss'])
parser.add_argument('--gp_loss_lambda',     default=75,                 type=float)
parser.add_argument('--wingloss_w',         default=10,                  type=int)
parser.add_argument('--wingloss_e',         default=2,                   type=int)

# resume training parameters
parser.add_argument('--resume_epoch',       default=0,                   type=int)
parser.add_argument('--resume_folder',      default='./weights/checkpoints/',  type=str)

# model saving parameters
parser.add_argument('--save_folder',        default='./weights/',        type=str)
parser.add_argument('--save_interval',      default=5,                 type=int)

# model setting
parser.add_argument('--hour_stack',         default=4,                   type=int)
parser.add_argument('--msg_pass',           default=True,                type=bool)
parser.add_argument('--GAN',                default=True,                type=bool)
parser.add_argument('--fuse_stage',         default=4,                   type=int)
parser.add_argument('--sigma',              default=1.0,                 type=float)
parser.add_argument('--theta',              default=1.5,                 type=float)
parser.add_argument('--delta',              default=0.8,                 type=float)

# evaluate parameters
parser.add_argument('--eval_epoch',         default=20,                 type=int)
parser.add_argument('--max_threshold',      default=0.1,                 type=float)
parser.add_argument('--norm_way',           default='inter_ocular',      type=str,
                    choices=['inter_pupil', 'inter_ocular', 'face_size'])
parser.add_argument('--eval_visual',        default=True ,               type=bool)
parser.add_argument('--save_img',           default=True,                type=bool)

def parse_args():
    result = parser.parse_args()

    assert result.resume_epoch < result.step_values[0]
    assert result.resume_epoch < result.max_epoch
    assert result.step_values[-1] < result.max_epoch
    return result