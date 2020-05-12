import torch.nn as nn
import tqdm

from utils import *
from utils.args import parse_args
from utils.dataset import ShapeDataset
from sklearn.decomposition import PCA


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
          '# Dataset:              ' + arg.dataset + '\n' +
          '# Dataset split:        ' + arg.split + '\n' +
          '# Dataset split source: ' + arg.split_source + '\n')
    if arg.resume_epoch > 0:
        print('# Resumed epoch:      ' + str(arg.resume_epoch))

    print('Creating networks ...')
    pca_model = create_model_pca(arg, devices, eval=False, inverse=arg.pca_inverse)
    suffix = '_pca_inverse' if arg.pca_inverse else '_pca'
    print('Creating networks done!')

    print('Loading dataset ...')
    trainset = ShapeDataset(arg, dataset=arg.dataset, split=arg.split, pca_components=arg.pca_components)
    trainset_source = ShapeDataset(arg, dataset=arg.dataset, split=arg.split_source, pca_components=arg.pca_components)
    print('Loading dataset done!')

    # evolving training
    print('Start training ...')
    shapes = np.concatenate((trainset.shapes, trainset_source.shapes), axis=0)

    pca = PCA(n_components=arg.pca_components, svd_solver='full')
    pca.fit(shapes)

    pca_model.load_parameters(pca.components_, pca.mean_, inverse=arg.pca_inverse)

    torch.save(pca_model.state_dict(), arg.save_folder + arg.dataset + '_' + arg.split + '+' + arg.split_source + suffix + '.pth')
    print('Training done!')
    print('done')


if __name__ == '__main__':
    arg = parse_args()
    main(arg)