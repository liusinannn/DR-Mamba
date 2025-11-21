
import argparse
import logging
import os
import sys
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
# from pygments.lexers.sql import name_between_backtick_re
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# from MLTMamba_SS2D import RSM_SS,DualEncoderHIF
from MLTmamba.losses import CEDiceLoss
# from RS3Mamba import RS3Mamba
# from DRmamba import RSM_SS
# from rs3mb import RS3Mamba
# from MLTMamba_r import DualEncoderHIF
# from vmamba import *
# from models.mamba_vision import *
from DRmamba_bf_pretrained_2 import DualEncoderHIF

from dataset import BasicDataset
# from utils.eval import eval_net
import matplotlib.pyplot as plt
import warnings

import torch.nn.functional as F

warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# dir_img = r'D:\GIS\LULC\3datasets(LULC)\LoveDA\Rural\image'
# dir_mask = r'D:\GIS\LULC\3datasets(LULC)\LoveDA\Rural\label'
#
#
# val_img = r'D:\GIS\LULC\3datasets(LULC)\LoveDA\Rural_test\image'
# val_mask = r'D:\GIS\LULC\3datasets(LULC)\LoveDA\Rural_test\label'

dir_img = r'D:\GIS\LULC\3datasets(LULC)\LandCover.ai\train\image'
dir_mask = r'D:\GIS\LULC\3datasets(LULC)\LandCover.ai\train\label'

val_img = r'D:\GIS\LULC\3datasets(LULC)\LandCover.ai\test\image'
val_mask = r'D:\GIS\LULC\3datasets(LULC)\LandCover.ai\test\label'

dir_checkpoint = r'D:\GIS\Deep_PSP_UNETF_CMTF_RS3\ckpt/'

def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-g", "--gpu_id", dest="gpu_id", metavar="G", type=int, default=0, help="GPU ID"
    )
    parser.add_argument(
        "-u",
        "--unet_type",
        dest="unet_type",
        metavar="U",
        type=str,
        default="v1",
        help="UNet type: v1/v2/v3",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="E",
        type=int,
        default=100,
        help="Number of epochs",
        dest="epochs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="B",
        type=int,
        nargs="?",
        default=4,
        help="Batch size",
        dest="batchsize",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        metavar ="LR",
        type=float,
        nargs="?",
        default=1e-5,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "-f",
        "--load",
        dest="load",
        type=str,
        default=False,
        help="Load model from a .pth file",
    )
    parser.add_argument(
        "-s",
        "--scale",
        dest="scale",
        type=float,
        default=1,
        help="Downscaling factor of the images",
    )
    parser.add_argument(
        "-v",
        "--validation",
        dest="val",
        type=float,
        default=0,
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument(
        "--val_img",
        dest="val_img",
        type=str,
        default=val_img,  # 验证集图像路径
        help="Validation images directory"
    )
    parser.add_argument(
        "--val_mask",
        dest="val_mask",
        type=str,
        default=val_mask,  # 验证集标签路径
        help="Validation masks directory"
    )

    return parser.parse_args()


def train_net(
    unet_type,
    net,
    device,
    epochs=100,
    batch_size=3,
    lr=0.01,
    val_percent=0.1,
    save_cp=True,
    img_scale=1,
    val_img_dir=None,  # 新增验证集路径参数
    val_mask_dir=None
):
    # 加载训练集
    train = BasicDataset(dir_img, dir_mask, img_scale)

    # 加载验证集
    val = BasicDataset(val_img_dir, val_mask_dir, img_scale)

    n_train = len(train)
    n_val = len(val)

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True
    )

    writer = SummaryWriter(comment=f"LR_{lr}_BS_{batch_size}_SCALE_{img_scale}")
    global_step = 0

    logging.info(
        f"""Starting training:
        Net type:       {'UNet'}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Dataset size:    {len(train)}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device}
        Images scaling:  {img_scale}"""
    )

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-2)

    # 2. 使用 CosineAnnealingLR 来实现从初始值到最小值的平滑衰减
    #    T_max: 衰减过程持续的 epoch 总数，我们设置为总的训练周期
    #    eta_min: 学习率的下限，即您希望最终衰减到的值 1e-7
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    scaler = GradScaler()

    # lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = CEDiceLoss()

    # 新增：记录每个epoch的平均训练和验证损失
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.8f}")
        net.train()
        total_train_loss = 0

        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            for batch in train_loader:
                imgs = batch["image"]
                true_masks = batch["mask"].squeeze(1)
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                optimizer.zero_grad()
                with autocast():
                    masks_pred = net(imgs)
                    # masks_pred_upsampled = F.interpolate(
                    #     masks_pred,
                    #     size=(512, 512),
                    #     mode='bilinear',
                    #     align_corners=False
                    # )
                    # print(masks_pred.shape)
                    # print(true_masks.shape)
                    loss = criterion(masks_pred, true_masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                pbar.update(imgs.shape[0])
                global_step += 1
                total_train_loss += loss.item()
                writer.add_scalar("Loss/train", loss.item(), global_step)

        # 计算本轮训练的平均损失并记录
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)
        scheduler.step()
        # 验证过程
        net.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device=device, dtype=torch.float32)
                true_masks = batch["mask"].squeeze(1).to(device=device, dtype=mask_type)
                with autocast():
                    masks_pred = net(imgs)
                    # masks_pred_upsampled = F.interpolate(
                    #     masks_pred,
                    #     size=(512, 512),
                    #     mode='bilinear',
                    #     align_corners=False
                    # )
                    loss = criterion(masks_pred, true_masks)
                total_val_loss += loss.item()

        # 计算本轮验证的平均损失并记录
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        writer.add_scalar("Loss/validation", val_loss, global_step)

        logging.info(
            f"Epoch {epoch + 1} finished! Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        if save_cp and (epoch + 1) % 5 == 0:
            try:
                os.makedirs(dir_checkpoint, exist_ok=True)
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(
                net.state_dict(),
                os.path.join(dir_checkpoint, f"CP_epoch{epoch + 1}_DRmamba_landcover.pth"),
            )
            logging.info(f"Checkpoint {epoch + 1} saved !")

    writer.close()

    # 绘制训练和验证损失图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print(torch.cuda.device_count())
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # print(torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    # torch.cuda.empty_cache()
    # 根据 unet_type 初始化不同的 UNet 模型
    net = DualEncoderHIF(5)
    # net = ()

    net.to(device=device)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    try:
        train_net(
            unet_type=args.unet_type,
            net=net,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            val_img_dir=args.val_img,  # 传递验证集路径
            val_mask_dir=args.val_mask
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        sys.exit(0)
