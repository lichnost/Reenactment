import torch.nn as nn
import tqdm

from utils import *
from utils.args import parse_args
from utils.dataset import GeneralDataset


def main(arg):
    log_writer = None
    if arg.save_logs:
        if not os.path.exists('./logs'):
            os.mkdir('./logs')
        log_writer = SummaryWriter()

    epoch = None
    devices = get_devices_list(arg)

    print('*****  Training PCA  *****')
    print('Training parameters:\n' +
          '# Dataset:            ' + arg.dataset + '\n' +
          '# Dataset split:      ' + arg.split + '\n' +
          '# Batchsize:          ' + str(arg.batch_size) + '\n' +
          '# Num workers:        ' + str(arg.workers) + '\n' +
          '# Use GPU:            ' + str(arg.cuda) + '\n' +
          '# Start lr:           ' + str(arg.lr) + '\n' +
          '# Max epoch:          ' + str(arg.max_epoch) + '\n' +
          '# Loss type:          ' + arg.loss_type + '\n' +
          '# Resumed model:      ' + str(arg.resume_epoch > 0))
    if arg.resume_epoch > 0:
        print('# Resumed epoch:      ' + str(arg.resume_epoch))

    print('Creating networks ...')
    pca = create_model_pca(arg, devices, eval=True)
    pca.eval()
    print('Creating networks done!')

    print('Loading dataset ...')
    trainset = GeneralDataset(arg, dataset=arg.dataset, split=arg.split)
    print('Loading dataset done!')

    dataloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, shuffle=arg.shuffle,
                                             num_workers=arg.workers, pin_memory=False)

    # evolving training
    print('Start Evaluating ...')
    for epoch in range(arg.resume_epoch, arg.max_epoch):
        forward_times_per_epoch, sum_loss_pca = 0, 0.

        if epoch in arg.step_values:
            optimizer_pca.param_groups[0]['lr'] *= arg.gamma

        for data in tqdm.tqdm(dataloader):
            forward_times_per_epoch += 1

            pic_affine_orig, pic_affine, pic_affine_orig_norm, gt_coords_xy, gt_heatmap, _, _, _ = data
            if arg.cuda:
                gt_coords_xy = gt_coords_xy.cuda(device=devices[0])
                pose_param = pose_param.cuda(device=devices[0])
            gt_coords_xy = gt_coords_xy.reshape(gt_coords_xy.shape[0], -1)


            optimizer_pca.zero_grad()
            pred_pose_param = pca(gt_coords_xy)
            loss_pca = criterion(pred_pose_param, pose_param)
            loss_pca.backward()
            optimizer_pca.step()

            sum_loss_pca += loss_pca.item()

        if (epoch+1) % arg.save_interval == 0:
            torch.save(pca.state_dict(), arg.save_folder + arg.dataset + '_pca_' + str(epoch+1) + '.pth')

        # if log_writer is not None:
        #     log_writer.add_scalar()

        print('\nepoch: {:0>4d} | loss_decoder: {:.10f}'.format(
            epoch,
            sum_loss_pca/forward_times_per_epoch,
        ))

    torch.save(pca.state_dict(), arg.save_folder + arg.dataset + '_pca_' + str(epoch+1) + '.pth')
    print('Training done!')
    # component = 0
    #
    # samples_in_bins = np.histogram(pose_params[:, component], dataset_pdb_numbins[dataset])
    # samples_num = samples_in_bins[0].shape[0]
    # for idx_bin in range(samples_num):
    #     start = samples_in_bins[1][idx_bin]
    #     end = samples_in_bins[1][idx_bin + 1]
    #
    #     start_indexes = np.argwhere(pose_params[:, 0] >= start)
    #     end_indexes = np.argwhere(pose_params[:, 0] < end) if idx_bin + 1 != samples_num else np.argwhere(pose_params[:, 0] <= end)
    #     start_indexes = start_indexes.reshape((start_indexes.shape[0]))
    #     end_indexes = end_indexes.reshape((end_indexes.shape[0]))
    #     indexes = np.intersect1d(start_indexes, end_indexes)
    #
    #     idx_item = indexes[randint(0, len(indexes) - 1)]
    #     line = annotations[idx_item]
    #     image = cv2.imread(arg.dataset_route[dataset] + line[-1])
    #
    #     shape = np.uint(shapes[:kp_num[dataset] * 2, idx_item].reshape((-1, 2), order='F'))
    #     for point in shape:
    #         draw_circle(image, tuple(point))
    #     show_img(image)

    print('done')


if __name__ == '__main__':
    arg = parse_args()
    main(arg)