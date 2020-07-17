from utils import *
from utils.args import parse_args
from flame.FLAME import get_flame_layer, render_images, random_texture, get_texture_model
import cv2, torch
from kornia import image_to_tensor, tensor_to_image, rgb_to_bgr, denormalize
from utils.dataset import DecoderDataset

def fit_flame(arg):
    log_writer = None
    if arg.save_logs:
        log_path = './logs/transformer_' + arg.dataset + '_' + arg.split_source + '2' + arg.split
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_writer = SummaryWriter(log_dir=log_path)
        def log(tag, scalar, step):
            log_writer.add_scalar(tag, scalar, step)
        def log_img(tag, img, step):
            log_writer.add_image(tag, img, step)
        def log_text(tag, text, step):
            log_writer.add_text(tag, text, step)
    else:
        def log(tag, scalar, step):
            pass
        def log_img(tag, img, step):
            pass
        def log_text(tag, text, step):
            pass


    devices = get_devices_list(arg)

    print('Creating FLAME model layer ...')
    flame = create_model_flame(arg, devices)

    print('Creating FLAME model layer done!')

    print('Loading dataset ...')
    trainset = DecoderDataset(arg, dataset=arg.dataset, split=arg.split)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size, drop_last=True,
                                             num_workers=arg.workers, pin_memory=True)
    print('Loading dataset done!')

    mean = torch.FloatTensor(means_color[arg.dataset][arg.split])
    std = torch.FloatTensor(stds_color[arg.dataset][arg.split])
    norm_min = (0 - mean) / std
    norm_max = (255 - mean) / std
    norm_range = norm_max - norm_min
    norm_range = torch.where(norm_range < 1e-6, torch.ones_like(norm_range), norm_range)

    iterator = iter(dataloader)
    input_images, input_images_norm, gt_coords_xy = next(iterator)

    show_img(np.uint8(tensor_to_image(rgb_to_bgr(denormalize(input_images_norm[0], mean, std)))), 'target')

    if arg.cuda:
        input_images = input_images.to(device=devices[0])
        input_images_norm = input_images_norm.to(device=devices[0])
        gt_coords_xy = gt_coords_xy.to(device=devices[0])

    input_images = denormalize(input_images, mean, std)



    # images = []
    # for path in arg.image_paths:
    #     images.append(image_to_tensor(cv2.imread(path)))
    # images = torch.stack(images)




if __name__ == '__main__':
    arg = parse_args()

    if not os.path.exists(arg.save_folder):
        os.mkdir(arg.save_folder)
    if not os.path.exists(arg.resume_folder):
        os.mkdir(arg.resume_folder)

    fit_flame(arg)