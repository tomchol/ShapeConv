import random
import torch
import cv2
import sys
import numpy as np
import tifffile

sys.path.append("/home/imag2/IMAG2_DL/imag_segmentation/ShapeConv/")

# from ShapeConv.rgbd_seg.dataloaders import build_dataloader
from ShapeConv.rgbd_seg.models import build_model
from ShapeConv.rgbd_seg.utils import Config, gather_tensor, load_checkpoint
from ShapeConv.rgbd_seg.datasets import build_dataset
from ShapeConv.rgbd_seg.transforms import build_transform
from ShapeConv.rgbd_seg.dataloaders.samplers import build_sampler
from ShapeConv.rgbd_seg.dataloaders import build_dataloader


def transform_output(output):
    x_shape, y_shape = output.shape
    new_output = np.zeros((x_shape, y_shape, 3), dtype=np.float32)
    list_classes = [[0, 0, 0], [0, 255, 0], [0, 55, 255],  [125, 55, 24], [0, 255, 125], [5, 155, 124], [255, 255, 125]]

    for classe in range(7):
        print(classe, np.count_nonzero(output == classe))
        new_output[output == classe] = list_classes[classe]

    return new_output


def build_dataloader_inter(cfg):
    transform = build_transform(cfg['transforms'])
    dataset = build_dataset(cfg['dataset'], dict(transform=transform))

    shuffle = cfg['dataloader'].pop('shuffle', False)
    sampler = build_sampler(False,
                            cfg['sampler'],
                            dict(dataset=dataset,
                                 shuffle=shuffle))

    dataloader = build_dataloader(False,
                                  1,
                                  cfg['dataloader'],
                                  dict(dataset=dataset,
                                       sampler=sampler))

    return dataloader


def resume(model, checkpoint, map_location='default'):
    checkpoint = load_checkpoint_inter(model, checkpoint,
                                       map_location=map_location)


def load_checkpoint_inter(model, filename, map_location='default', strict=True):
    print('Load checkpoint from {}'.format(filename))

    if map_location == 'default':
        device_id = torch.cuda.current_device()
        map_location = lambda storage, loc: storage.cuda(device_id)

    return load_checkpoint(model, filename, map_location, strict)


def read_tiff16(fn):
    img = tifffile.imread(fn)
    if img.dtype == np.uint8:
        depth = 8
    elif img.dtype == np.uint16:
        depth = 16
    elif img.dtype == np.float32:
        return img

    else:
        print("Warning: unsupported data type {}. Assuming 16-bit.", img.dtype)
        depth = 16

    return (img * (1.0 / (2 ** depth - 1))).astype(np.float32)


def main(cfg_path):
    cfg = Config.fromfile(cfg_path)

    train_cfg = cfg['train']
    inference_cfg = cfg['inference']

    # build model
    model = build_model(inference_cfg['model'])
    model.eval()

    # chargement du model entraine
    if train_cfg.get('resume'):
        resume(model, train_cfg['resume'])

    val_dataloader = build_dataloader_inter(
        train_cfg['data']['val'])

    # path = "/mnt/HDD_4T/Disparity_map_dataset/training/dataset_6/keyframe_4/"
    # image = cv2.cvtColor(cv2.imread(path + "Left_Image.png"), cv2.COLOR_BGR2RGB)
    # depth = read_tiff16(path + "left_depth_map.tiff")
    # image_depth = torch.zeros((1, 4, 1024, 1280))
    # image_depth[0][0][:] = torch.from_numpy(image[:, :, 0])
    # image_depth[0][1][:] = torch.from_numpy(image[:, :, 1])
    # image_depth[0][2][:] = torch.from_numpy(image[:, :, 2])
    # image_depth[0][3][:] = torch.from_numpy(depth[:, :, 2])
    #
    # with torch.no_grad():
    #     output = model(image_depth)
    #     output = output.softmax(dim=1)
    #     _, output = torch.max(output, dim=1)
    #
    #     output = gather_tensor(output)
    #
    #     image_png = cv2.cvtColor(image_depth[0][:3].transpose(0, 2).transpose(0, 1).numpy(), cv2.COLOR_BGR2RGB)
    #     cv2.imwrite("/home/imag2/IMAG2_DL/imag_segmentation/ShapeConv/results/image.png", image_png)
    #     output_png = transform_output(output[0].numpy())
    #     added_image = cv2.addWeighted(image_png, 1, output_png, 1, 0)
    #     cv2.imwrite("/home/imag2/IMAG2_DL/imag_segmentation/ShapeConv/results/output.png", added_image)

    i = random.randint(0, len(val_dataloader))
    #i = 52 + 250 * 0
    print("val", i//250 + 1, "frame", i % 250)
    for idx, (image, mask) in enumerate(val_dataloader):
        if idx == i:
            with torch.no_grad():
                output = model(image)
                output = output.softmax(dim=1)
                _, output = torch.max(output, dim=1)

                output = gather_tensor(output)
                mask = gather_tensor(mask)

                image_png = cv2.cvtColor(image[0][:3].transpose(0, 2).transpose(0, 1).numpy(), cv2.COLOR_BGR2RGB)
                cv2.imwrite("/home/imag2/IMAG2_DL/imag_segmentation/ShapeConv/results/image.png", image_png)
                cv2.imwrite("/home/imag2/IMAG2_DL/imag_segmentation/ShapeConv/results/mask.png", mask[0].numpy())
                output_png = transform_output(output[0].numpy())
                added_image = cv2.addWeighted(image_png, 1, output_png, 1, 0)
                cv2.imwrite("/home/imag2/IMAG2_DL/imag_segmentation/ShapeConv/results/output.png", added_image)
            break


if __name__ == '__main__':
    main("/home/imag2/IMAG2_DL/imag_segmentation/ShapeConv/configs/nyu/nyu6_deeplabv3plus_vis.py")
