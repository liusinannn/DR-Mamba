import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchsummary as summary
from PIL import Image
from osgeo import gdal
from torchvision import transforms
# from generators.unet import UNet
# from generators.deeplab.deeplabv3_plus import DeepLab
# from utils.data_vis import plot_img_and_mask
from dataset import BasicDataset
# from segformer_pytorch import Segformerb5
# from MLTMamba_SS2D import RSM_SS,DualEncoderHIF
# from MLTMamba_r import RSM_SS,DualEncoderSeg,DualEncoderHIF
# from RS3Mamba import RS3Mamba
from DRMamba import DualEncoderHIF
Image.MAX_IMAGE_PIXELS = None


def predict_img(net, full_img, device, img_scale=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, img_scale))
    img = img.unsqueeze(0)[:, :3, :, :]
    img = img.to(device=device, dtype=torch.float32)

    # color_map = {
    #     0: (0, 0, 0),  # 未分类 - 黑色（或nodata）
    #     1: (255, 255, 255),  # 背景 - 浅灰
    #     2: (255, 0, 0),  # 建筑 - 鲜红色
    #     3: (255, 255, 0),  # 道路 - 灰色
    #     4: (0, 0, 255),  # 水 - 蓝色
    #     5: (159, 129, 183),  # 裸土 - 紫色
    #     6: (0, 255, 0),  # 林地 - 绿色
    #     7: (255, 195, 128),  # 农业 - 橙色
    #
    # }
    color_map = {
        # 0: (255, 195, 128),  # 农业 - 橙色
        0: (255, 255, 255),  # 农业 - 橙色
        1: (255, 0, 0),  # 建筑 - 红色
        2: (0, 255, 0),  # 林地 - 绿色
        3: (0, 0, 255),  # 水 - 蓝色
        4: (255, 255, 0)  # 道路 - 黄色
    }

    with torch.no_grad():
        output = net(img)

        probs = F.softmax(output, dim=1)
        # if net.n_classes > 1:
        #     probs = F.softmax(output, dim=1)
        # else:
        #     probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        predictions = probs.argmax(dim=0).cpu().numpy()

        # 创建一个空的彩色图像
        color_image = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)

        # 用颜色填充彩色图像
        for label, color in color_map.items():
            color_image[predictions == label] = color

        # tf = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(full_img.size[1]),
        #         transforms.ToTensor(),
        #     ]
        # )
        # probs = tf(probs.cpu())
        # full_mask = probs.squeeze().cpu().numpy()
    return color_image


def get_args():
    class Args:
        gpu_id = 0
        unet_type = (
            "v1"  # 'v1', 'v2', 'v3', 'UNet3Plus_DeepSup', 或 'UNet3Plus_DeepSup_CGM'
        )
        model = r"D:\GIS\DR-Mamba_XR\ckpt\CP_epoch45_DRMamba_land.pth"
        input_folder = r"D:\GIS\LULC\3datasets(LULC)\LandCover.ai\test\image"  # 指定输入图像所在的文件夹路径
        # output_folder = r"D:\GIS\LULC\aaaaaaaaaaaaaaaa\DR-mamba\loveda"  # 指定输出图像的文件夹路径
        # input_folder = r"D:\GIS\LULC\3datasets(LULC)\LoveDA\Rural_test\image"  # 指定输入图像所在的文件夹路径
        output_folder = r"D:\GIS\LULC\aaaaaaaaaaaaaaaa\DR-mamba\landcover.ai_final"  # 指定输出图像的文件夹路径
        viz = True
        no_save = False
        scale = 1

    return Args()


def get_output_filenames(args):
    in_files = [
        os.path.join(args.input_folder, f)
        for f in os.listdir(args.input_folder)
        if f.endswith((".jpg", ".png", ".tif"))
    ]
    out_files = []

    for f in in_files:
        pathsplit = os.path.splitext(f)
        out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def get_georeference_info(input_folder):
    input_files = [
        f for f in os.listdir(input_folder) if f.endswith((".jpg", ".png", ".tif"))
    ]
    input_file = os.path.join(
        input_folder, input_files[0]
    )  # 使用第一个输入文件获取空间参考信息
    input_dataset = gdal.Open(input_file)
    geotransform = input_dataset.GetGeoTransform()
    projection = input_dataset.GetProjection()
    return geotransform, projection


def save_mask_with_georeference(mask, output_path, geotransform, projection):
    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(
        output_path, mask.shape[1], mask.shape[0], 1, gdal.GDT_Byte
    )
    output_dataset.SetGeoTransform(geotransform)
    output_dataset.SetProjection(projection)
    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(mask)
    output_band.FlushCache()
    output_dataset = None


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # net = RS3Mamba(num_classes=8)
    net = DualEncoderHIF(5)

    logging.info("Loading model {}".format(args.model))
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")

    in_files = [
        os.path.join(args.input_folder, f)
        for f in os.listdir(args.input_folder)
        if f.endswith((".jpg", ".png", ".tif"))
    ]
    for i, fn in enumerate(in_files):
        geotransform, projection = get_georeference_info(os.path.dirname(fn))
        logging.info("\nPredicting image {} ...".format(fn))
        img = Image.open(fn)
        mask = predict_img(
            net=net,
            full_img=img,
            img_scale=args.scale,
            device=device,
        )
        color_image_pil = Image.fromarray(mask)
        color_image_pil.save(os.path.join(args.output_folder, os.path.basename(fn)))

        # 将预测结果映射到 0-255 范围
        # mask[mask >= 0.3] = 1
        # mask[mask < 0.3] = 0
        # mask = (mask * 255).astype(np.uint8)
        #
        # if not args.no_save:
        #     out_fn = os.path.join(args.output_folder, os.path.basename(fn))
        #     save_mask_with_georeference(mask, out_fn, geotransform, projection)
        #     logging.info("Mask saved to {}".format(out_fn))
        #
        # if args.viz:
        #     logging.info(
        #         "Visualizing results for image {}, close to continue ...".format(fn)
        #     )
