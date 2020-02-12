import os
import tqdm
import torch
from torch.utils.data import DataLoader
from utils.args import parse_args
from utils.dataset import OriginalImageDataset
from kornia import rgb_to_grayscale


def main(arg):
    print('*****  Dataset info  *****')
    print('Evaluating parameters:\n' +
          '# Dataset:            ' + arg.dataset + '\n' +
          '# Dataset split:      ' + arg.split + '\n')
    dataset = OriginalImageDataset(arg, arg.dataset, arg.split)

    cnt = 0
    mean_color = torch.zeros(3)
    std_color = torch.zeros(3)

    mean_gray = torch.zeros(1)
    std_gray = torch.zeros(1)

    for data in tqdm.tqdm(dataset):
        mean_color += torch.mean(data, dim=(1, 2))
        std_color += torch.std(data, dim=(1, 2))

        data_gray = rgb_to_grayscale(data)
        mean_gray += torch.mean(data_gray)
        std_gray += torch.std(data_gray)

        cnt += 1

    mean_color = mean_color / cnt
    std_color = std_color / cnt

    mean_gray = mean_gray / cnt
    std_gray = std_gray / cnt

    print('Means color: ' + str(mean_color))
    print('Stds color: ' + str(std_color))
    print('Means gray: ' + str(mean_gray))
    print('Stds gray: ' + str(std_gray))

if __name__ == '__main__':
    arg = parse_args()
    main(arg)
