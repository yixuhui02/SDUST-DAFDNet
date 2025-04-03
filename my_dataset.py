########################################################################################################################
# 数据预处理模块
########################################################################################################################

import os
import torch.utils.data as data
from PIL import Image
import numpy as np
from osgeo import gdal
from torchvision import transforms
class VOCSegmentation(data.Dataset):
    def __init__(self, root, transforms__=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()

        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'images')
        edge_dir = os.path.join(root, 'edge')
        mask_dir = os.path.join(root, 'labels')
        txt_path = os.path.join(root, txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)

        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        #self.images = [os.path.join(image_dir, x + ".tif") for x in file_names]
        #self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]

        self.images = [os.path.join(image_dir, x.split('.')[0] + ".tif") for x in file_names]
        #self.edge = [os.path.join(edge_dir, x.split('.')[0] + ".png") for x in file_names]
        self.masks = [os.path.join(mask_dir, x.split('.')[0]  + ".tif") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms__
        self.transforms_ = transforms.ToTensor()

        # self.transform___ = transforms.Compose([
        #
        #     transforms.ToTensor(),
        #     #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     transforms.Resize((512, 512))                                                                   # 尺寸修改
        #
        # ])


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img_ds = gdal.Open(self.images[index], gdal.GA_ReadOnly)
        # 读取图像的波段数、宽度和高度
        bands = img_ds.RasterCount
        width = img_ds.RasterXSize
        height = img_ds.RasterYSize


        image_data = np.zeros((height, width, bands), dtype=np.float32)


        for band in range(bands):
            band_data = img_ds.GetRasterBand(band + 1).ReadAsArray().astype(np.float32)

            band_data = band_data / (2 ** 16 - 1)
            image_data[:, :, band] = band_data

        img_ds = None
        # image_data = torch.from_numpy(image_data)
        img = self.transforms_(image_data)

        #img = Image.open(self.images[index]).convert('RGB')
        #print(self.masks[index] , self.edge[index])
        target = Image.open(self.masks[index])
        #edge = Image.open(self.edge[index])
        target1 = target
        # def count_unique_values(matrix):
        #     unique_values = {}  # 使用字典来存储不重复的数值及其像素点个数
        #     for row in matrix:
        #         for num in row:
        #             if num in unique_values:
        #                 unique_values[num] += 1
        #             else:
        #                 unique_values[num] = 1
        #     print("数值类别及其像素点个数：")
        #     for value, count in unique_values.items():
        #         print(f"数值 {value}: {count} 个像素点")
        #     return unique_values

        #count_unique_values(np.array(target))



        if self.transforms is not None:
            img, target = self.transforms(img, target)
            #edge = self.transform___(edge)
        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):

    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs



