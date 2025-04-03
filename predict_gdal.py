########################################################################################################################
# 测试脚本
########################################################################################################################

import os
import time
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from osgeo import gdal
from src import bsmsm
from src.deeplabv3_plus import DeepLab

from src.unet_model import  UNet
import torch.nn.functional as F
import transforms as T
from torchvision import transforms
from tqdm import tqdm

import cv2
from src import pbfam
from loss_function import get_label_to_color
class GRID1:

    def load_image(self, filename):

        # img_ds = gdal.Open(filename, gdal.GA_ReadOnly)
        # # 读取图像的波段数、宽度和高度
        # bands = img_ds.RasterCount
        # width = img_ds.RasterXSize
        # height = img_ds.RasterYSize
        #
        # # 初始化一个数组来存储图像数据
        # image_data = np.zeros((height, width, bands), dtype=np.float32)
        #
        # # 逐个波段读取并存储数据
        # for band in range(bands):
        #     band_data = img_ds.GetRasterBand(band + 1).ReadAsArray().astype(np.float32)
        #     # 将图像数据缩放到0到1之间

        #     image_data[:, :, band] = band_data

        image = gdal.Open(filename,gdal.GA_ReadOnly)

        img_width = image.RasterXSize
        img_height = image.RasterYSize

        img_geotrans = image.GetGeoTransform()
        img_proj = image.GetProjection()
        img_data = image.ReadAsArray(0, 0, img_width, img_height )

        #------------------------------------
        def count_unique_values(matrix):
            unique_values = set()  # 使用集合来存储不重复的数值
            for row in matrix:
                for num in row:
                    unique_values.add(num)
            print(unique_values)
            print("total_num_px:" ,len(unique_values))
            return len(unique_values)

        #count_unique_values(np.array(img_data))
        #---------------------------------------
        del image

        return img_proj, img_geotrans, img_data , img_width , img_height

    def write_image(self, filename, img_proj, img_geotrans, img_data):

        if 'int8' in img_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(img_data.shape) == 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape

        driver = gdal.GetDriverByName('GTiff')
        image = driver.Create(filename, img_width, img_height, img_bands, datatype)

        image.SetGeoTransform(img_geotrans)
        image.SetProjection(img_proj)

        if img_bands == 1:
            image.GetRasterBand(1).WriteArray(img_data)
        else:
            for i in range(img_bands):
                image.GetRasterBand(i + 1).WriteArray(img_data[i])

        del image


class GRID:

    def load_image(self, filename):
        image = gdal.Open(filename)

        bands = image.RasterCount
        img_width = image.RasterXSize
        img_height = image.RasterYSize

        img_geotrans = image.GetGeoTransform()
        img_proj = image.GetProjection()
        img_data = image.ReadAsArray(0, 0, img_width, img_height)

        image_data = np.zeros((img_height, img_width, bands), dtype=np.float32)

        # 逐个波段读取并存储数据
        for band in range(bands):
            band_data = image.GetRasterBand(band + 1).ReadAsArray().astype(np.float32)
            # 将图像数据缩放到0到1之间
            band_data = band_data / (2 ** 16 - 1)
            image_data[:, :, band] = band_data
        del image

        return img_proj, img_geotrans, image_data

    def write_image(self, filename, img_proj, img_geotrans, img_data):
        # if 'int8' in img_data.dtype.name:
        #     datatype = gdal.GDT_Byte
        # elif 'int16' in img_data.dtype.name:
        #     datatype = gdal.GDT_UInt16
        # else:
        #     datatype = gdal.GDT_Float32
        datatype = gdal.GDT_Byte

        if len(img_data.shape) == 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape

        driver = gdal.GetDriverByName('GTiff')
        image = driver.Create(filename, img_width, img_height, img_bands, datatype)

        image.SetGeoTransform(img_geotrans)
        image.SetProjection(img_proj)

        if img_bands == 1:
            image.GetRasterBand(1).WriteArray(img_data)
        else:
            for i in range(img_bands):
                image.GetRasterBand(i + 1).WriteArray(img_data[i])

        del image

def detect_image(image , num , model , device):




    tt = transforms.ToTensor()
    image2 =  tt(image)
    img = torch.unsqueeze(image2, dim=0)

    model.eval()
    with torch.no_grad():

        t_start = time.time()
        output = model(img.to(device))
        t_end = time.time()
        print("inference time: {}".format(t_end - t_start))
        output = output['out']

        pr = torch.squeeze(output, dim=0)
        pr = F.softmax(pr, dim=0).cpu().numpy()
        pr = pr.argmax(axis=0)

        pr = np.array(pr)
        #pr = np.where(pr == 0, 255, pr)

        pr_image = Image.fromarray(pr.astype('uint8'), 'L')

        file_name = './crop_combine/eval/predict/' + str(num)
        file_extension = '.tif'

        # 保存灰度图像
        pr_image.save('{}{}'.format(file_name, file_extension))


    return 1






def main():

    IMAGE_DIR = './crop_combine/eval/data/image'                                            # 图像文件夹(三通道)
    #path_out = './data/test_output/'                                                                                          # 输出文件夹
    txt_path =  './crop_combine/eval/data/eval.txt'


    with open(os.path.join(txt_path), "r") as f:
        file_names = [x.strip()for x in f.readlines() if len(x.strip()) > 0]

    # 128 用这个
    patch_list = [int(i.split('.')[0]) for i in file_names]
    patch_list = sorted(patch_list)
    patch_list = [str(i) for i in patch_list]

    #600 用这个
    # patch_list = [i.split('.')[0] for i in file_names]
    # patch_list = sorted(patch_list)
    # patch_list = [str(i) for i in patch_list]



    images_list = [os.path.join(IMAGE_DIR, x.split('.')[0] + ".tif") for x in patch_list]
    run = GRID()
    #count = os.listdir(IMAGE_DIR)

    classes = 5                                                                                                       # 类别数


    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = pbfam(num_classes=classes ,Regularization = 0.001)



    weights_path = './save_weights/best.pth'  # 权重路径
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    weights_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(weights_dict)
    model.to(device)

    for i,path in enumerate(tqdm(images_list)):
        #path = os.path.join(IMAGE_DIR, count[i])

        proj, geotrans, data = run.load_image(path)
        r_image = detect_image(data , i ,model ,device)

        #run.write_image(path_out + '{}.tif'.format(str(count[i])), proj, geotrans, r_image)

    ori_label_path = './crop_combine/train1.tif'  # 输入数据
    rgb = False


    patch_label_path = './crop_combine/eval/predict'
    output_path = './crop_combine/eval/temp/'



    # patch_label_path = './CAM/feature_vis'
    # output_path = './combine_cam/'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    patch_name_list = [i for i in os.listdir(patch_label_path)]
    patch_list = [int(i.split('.')[0]) for i in patch_name_list]
    patch_list = sorted(patch_list)
    patch_list = [str(i) for i in patch_list]
    new_patch_label_path = [os.path.join(patch_label_path, i.split('.')[0] + ".tif") for i in patch_list]
    run = GRID1()
    proj, geotrans, data, label_width, label_height = run.load_image(ori_label_path)

    patch_size_w = 128
    patch_size_h = 128
    if rgb:
        predict_label = np.full((label_height, label_width, 3), 255)
    else:
        predict_label = np.full((label_height, label_width), 255)

    num = 0
    for i in tqdm(range(label_height // patch_size_h)):
        for j in range(label_width // patch_size_w):
            # if num < len(new_patch_label_path):
            patch_label = np.array(Image.open(new_patch_label_path[num]))

            if rgb:
                predict_label[i * patch_size_h:(i + 1) * patch_size_h, j * patch_size_w:(j + 1) * patch_size_w,
                :] = patch_label
            else:
                predict_label[i * patch_size_h:(i + 1) * patch_size_h,
                j * patch_size_w:(j + 1) * patch_size_w] = patch_label

            # predict_label[j * patch_size_w:(j + 1) * patch_size_w,i * patch_size_h:(i + 1) * patch_size_h] = patch_label
            num = num + 1
    if rgb:
        predict_label = Image.fromarray(predict_label.astype('uint8'), 'RGB')
    else:
        predict_label = Image.fromarray(predict_label.astype('uint8'), 'L')


    file_name = output_path + 'predict_out31'
    file_extension = '.tif'

    # 保存灰度图像
    predict_label.save('{}{}'.format(file_name, file_extension))

    is_label = False

    predict_dir='./crop_combine/eval/temp/'
    save_dir = './crop_combine/eval/combine/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    predict_name = [i for i in os.listdir(predict_dir)]

    for  i in tqdm(predict_name):
        path = os.path.join(predict_dir , i)

        gray_image = Image.open(path)
        color_image = Image.new('RGB', (gray_image.size[0], gray_image.size[1]), (0, 0, 0))

        for y in range(gray_image.size[1]):
            for x in range(gray_image.size[0]):

                label = str(gray_image.getpixel((x, y)))

                color = get_label_to_color().get(label, (0, 0, 0))

                color_image.putpixel((x, y), color)

        color_image.save("{}{}".format(save_dir,i) )
        print('img save->./crop_combine/eval/combine')


if __name__ == '__main__':
    main()
