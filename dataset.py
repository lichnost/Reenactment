import torch.utils.data as data
from utils import get_annotations_list, get_item_from


class GeneralDataset(data.Dataset):

    def __init__(self, arg, dataset='WFLW', split='train'):
        self.arg = arg
        self.dataset_route = arg.dataset_route
        self.dataset = dataset
        self.split = split
        self.list = get_annotations_list(self.dataset_route, dataset, split, arg.crop_size, ispdb=arg.PDB)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        return get_item_from(self.dataset_route, self.dataset, self.split, self.list[item], self.arg.crop_size, self.arg.RGB, self.arg.sigma, self.arg.trans_ratio, self.arg.rotate_limit, self.arg.scale_ratio, self.arg.scale_horizontal, self.arg.scale_vertical)
