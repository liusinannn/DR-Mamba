import os
import sys
import logging
import torch
import numpy as np

import cv2
from os.path import splitext
from os import listdir
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')


    def __len__(self):
        return len(self.ids)


    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        # if unet_type != 'v3':
        #     pil_img = pil_img.resize((newW, newH))
        # else:
        #     new_size = int(scale * 640)
        #     pil_img = pil_img.resize((new_size, new_size))

        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans


    def __getitem__(self, i):
        idx = self.ids[i]
        # mask_file = glob(self.masks_dir + idx + '.tif')
        # img_file = glob(self.imgs_dir + idx + '.tif')
        mask_file = glob(self.masks_dir + '\\' + idx + '.*')
        img_file = glob(self.imgs_dir + '\\' + idx + '.*')

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0]).convert('RGB')

        # assert img.size == mask.size, f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        transform = transforms.Compose([
            transforms.PILToTensor()  # 将 PIL 图像或 numpy 数组转换为张量，并将像素值归一化到 [0, 1]
        ])
        # 将图片转换为张量
        img = self.preprocess(img, 1)
        mask = transform(mask)

        return {'image': img, 'mask': mask}

if __name__ == '__main__':
    unet_type = 'v1'
    dir_img = r'E:\Si-nanLiu\pythonproject1\dataset\LULC\part\image'
    dir_mask = r'E:\Si-nanLiu\pythonproject1\dataset\LULC\part\label'
    val_percent = 0.1
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = BasicDataset(dir_img, dir_mask, 1)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=25, pin_memory=True
    )

    for batch in train_loader:
        imgs = batch["image"]
        true_masks = batch["mask"].squeeze(1)
        imgs = imgs.to(device=device, dtype=torch.float32)
        mask_type = torch.long
        true_masks = true_masks.to(device=device, dtype=mask_type)

        print(imgs.shape)
        print(true_masks.shape)


        break