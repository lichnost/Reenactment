import torch.utils.data as data
from utils import get_annotations_list, get_item_from
import cv2
from kornia import image_to_tensor, bgr_to_rgb


class SimpleDataset(data.Dataset):

    def __init__(self, arg, dataset, split):
        self.arg = arg
        self.dataset = dataset
        self.split = split
        self.list = get_annotations_list(self.arg.dataset_route, dataset, split, arg.crop_size, ispdb=arg.PDB)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        img = cv2.imread(self.arg.dataset_route[self.dataset] + self.list[item][-1])
        img = image_to_tensor(img)
        img = bgr_to_rgb(img)
        return img


class GeneralDataset(data.Dataset):

    def __init__(self, arg, dataset, split):
        self.arg = arg
        self.dataset = dataset
        self.split = split
        self.type = arg.type
        self.list = get_annotations_list(self.arg.dataset_route, dataset, split, arg.crop_size, ispdb=arg.PDB)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        return get_item_from(self.arg.dataset_route, self.dataset, self.split, self.type, self.list[item], self.arg.crop_size,
                             self.arg.RGB, self.arg.sigma, self.arg.trans_ratio, self.arg.rotate_limit,
                             self.arg.scale_ratio, self.arg.scale_horizontal, self.arg.scale_vertical)