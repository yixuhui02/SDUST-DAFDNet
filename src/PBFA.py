########################################################################################################################
# Unet结构，基于resnet50主干网络
########################################################################################################################

from collections import OrderedDict
from typing import Dict
import matplotlib.pyplot as plt
import torch
from .BSSA import SBA
import torch.nn as nn
from torch import Tensor
from .PSMF import PSFM
import math
from timm.models.layers import DropPath
from .backbone_resnet import resnet50
from .Unet_decode import Up, OutConv
from src.cfb import EVCBlock
import numpy as np
import cv2
import os
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn import functional as F

# from ..old_src.cfb import EVCBlock
# from ..old_src.EMA import EMA

# class channel_attention(nn.Module):     #通道注意力模块
#     def __init__(self,channel , ratio=16 ):
#         super(channel_attention, self).__init__()
#
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel//ratio ,False),
#             nn.ReLU(),
#             nn.Linear(channel//ratio , channel ,False),
#
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self ,x):
#         b,c,h ,w = x.size()
#         max_pool = self.max_pool(x).view([b,c])
#         avg_pool = self.avg_pool(x).view([b,c])
#
#         max_pool = self.fc(max_pool)
#         avg_pool = self.fc(avg_pool)
#
#         out = max_pool+avg_pool
#         out = self.sigmoid(out).view([b,c,1,1])
#
#         return out*x
#
#
# class spatial_attention(nn.Module):     #空间注意力
#     def __init__(self):
#         super(spatial_attention, self).__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_pool = torch.max(x, dim=1, keepdim=True)[0]
#         avg_pool = torch.mean(x, dim=1, keepdim=True)
#         y = torch.cat([max_pool, avg_pool], dim=1)
#         y = self.conv(y)
#         return x * self.sigmoid(y)

# # RGM关系引导模块：对编码器分支和解码器分支进行重新编码，从而获得信息增强的特征，提高边缘像素的分割能力。
# class RGM(nn.Module):
#     """Construct the embeddings from patch, position embeddings.
#     """
#     def __init__(self,in_channels,out_channels):
#         super(RGM, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
#         self.root = nn.Sequential(   # 定义了一个序列，即卷积、批归一化和激活函数。
#             nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(out_channels),
#             nn.SiLU(),
#         )
#
#     def forward(self, EM, x):
#         EM = self.conv1(EM)
#         _, _, h, w = x.size()  # 获取输入 x 的尺寸，其中 h 和 w 分别表示输入的高度和宽度。
#         EM = F.interpolate(EM, size=(h, w), mode='bilinear', align_corners=True)  #  使用 F.interpolate 函数对 EM 进行上采样，调整其大小为 (h, w)，采用双线性插值方法，保持角点的对齐性。
#         x = torch.cat([x * EM, x], dim=1)  # 将 x 和经过上采样后的 EM 拼接在一起，沿着通道维度 dim=1 进行拼接。这种操作可能用于将上采样后的特征与原始特征进行融合。
#         x = self.root(x)
#         return x








#  添加模块——多路自注意力模块1
class AdaptiveAttention(nn.Module):
    def __init__(self, in_channels, out_channels, height):  # 256，256，32
        super(AdaptiveAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 定义自适应注意力的全连接层
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma 是一个可学习参数，用于加权输入和注意力输出之间的比例。
        self.conv1 = nn.Conv1d(height, height, kernel_size=3, stride=1, padding=1, bias=False)  # 一维卷积
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False),
        )
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
        self.conv_1x1= nn.Conv2d(in_channels*3, out_channels, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3,  bias=False)

    def forward(self, x):
        batch_size, _, height, width = x.size()  # 返回输入张量 x 的维度信息。
        # 计算query、key和value
        # F.adaptive_avg_pool2d函数会将输入self.conv_3x3(x)进行自适应平均池化，使得输出的尺寸为 [height, 1]，其中 height 是根据具体情况决定的高度值。
        query = self.channel_attention(F.adaptive_avg_pool2d(self.conv_3x3(x), [height, 1]))
        key = self.channel_attention(F.adaptive_avg_pool2d(x, [1, width]))

        value1 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(x, [height//2, width//2]))))   # 2,6,8
        value1 = F.interpolate(value1, size=x.size()[2:], mode='bilinear', align_corners=True)  # 双线性插值，将其调整到与原始输入x相同的高和宽

        value2 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(x, [height//6, width//6]))))
        value2 = F.interpolate(value2, size=x.size()[2:], mode='bilinear', align_corners=True)

        value3 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(x, [height//8, width//8]))))
        value3 = F.interpolate(value3, size=x.size()[2:], mode='bilinear', align_corners=True)

        value = torch.cat([value1, value2, value3], dim=1)
        value = self.conv_1x1(value)
        # 计算注意力分数
        attention_scores = torch.matmul(query, key)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.out_channels//2, dtype=torch.float32))
        # 计算注意力权重
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        # 使用注意力权重对value进行加权平均
        attended_value = torch.matmul(attention_weights, value)
        # 使用注意力机制融合原始输入和加权平均后的值
        output = x + self.gamma * attended_value
        return output

# 定义了一个卷积序列的类
class Conv2d_r(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=3,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            # bias=not (use_batchnorm),
        )
        relu =nn.SiLU()
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2d_r, self).__init__(conv, bn, relu)
# 多感受野卷积模块，由三个部分组成，每个部分包含了三个串联的卷积层和一个DropPath层，用于增加模型的非线性和稀疏性。
class MultiReceptiveFieldConvModule(nn.Module):
    def __init__(self, in_channels, out_channels,dp_rate,p,d,a):
        super(MultiReceptiveFieldConvModule, self).__init__()
        out_r = out_channels//16
        # 定义了三个卷积对象，每一个中包含三个串联的3x3的卷积
        self.extra_conv1 = nn.Sequential(
            Conv2d_r(in_channels, out_r, kernel_size=3, padding=p[0], dilation=p[0]),
            Conv2d_r(out_r, out_r, kernel_size=3, padding=d[0], dilation=d[0]),
            Conv2d_r(out_r, out_channels, kernel_size=3, padding=a[0], dilation=a[0]),
            DropPath(dp_rate)
        )
        self.extra_conv2 = nn.Sequential(
            Conv2d_r(in_channels, out_r, kernel_size=3, padding=p[1], dilation=p[1]),
            Conv2d_r(out_r, out_r, kernel_size=3, padding=d[1], dilation=d[1]),
            Conv2d_r(out_r, out_channels, kernel_size=3, padding=a[1], dilation=a[1]),
            DropPath(dp_rate)
        )
        self.extra_conv3 = nn.Sequential(
            Conv2d_r(in_channels, out_r, kernel_size=3, padding=p[2], dilation=p[2]),
            Conv2d_r(out_r, out_r, kernel_size=3, padding=d[2], dilation=d[2]),
            Conv2d_r(out_r, out_channels, kernel_size=3, padding=a[2], dilation=a[2]),
            DropPath(dp_rate)
        )

    def forward(self, x):
        # 第一个3x3卷积
        extra_out1 = self.extra_conv1(x)
        # 两个额外的3x3卷积
        extra_out2 = self.extra_conv2(x+extra_out1)
        extra_out3 = self.extra_conv3(x+extra_out1)
        output = extra_out1+extra_out2+extra_out3
        return output
# # 添加模块——多路自注意力模块2
# class AdaptiveAttention(nn.Module):
#     def __init__(self, in_channels, out_channels, height):
#         super(AdaptiveAttention, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         # 定义自适应注意力的全连接层
#         self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#         self.gamma = nn.Parameter(torch.zeros(1))  # gamma 是一个可学习参数，用于加权输入和注意力输出之间的比例。
#         self.conv1 = nn.Conv1d(height, height, kernel_size=3, stride=1, padding=1, bias=False)  # 一维卷积
#
#         self.channel_attention = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // 16, in_channels, 1, bias=False),
#         )
#         self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
#         self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
#         self.conv_1x1= nn.Conv2d(in_channels*3, out_channels, kernel_size=1, stride=1)
#         self.conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3,  bias=False)
#
#     def forward(self, x, y):
#         batch_size, _, height, width = x.size()  # 返回输入张量 x 的维度信息。
#         batch_size_y, _, height_y, width_y = y.size()
#         # 计算query、key和value
#         # F.adaptive_avg_pool2d函数会将输入self.conv_3x3(x)进行自适应平均池化，使得输出的尺寸为 [height, 1]，其中 height 是根据具体情况决定的高度值。
#         query = self.channel_attention(F.adaptive_avg_pool2d(self.conv_3x3(x), [height, 1]))
#         key = self.channel_attention(F.adaptive_avg_pool2d(x, [1, width]))
#
#         value1 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(y, [height_y//2, width_y//2]))))   # 2,6,8
#         value1 = F.interpolate(value1, size=y.size()[2:], mode='bilinear', align_corners=True)  # 双线性插值，将其调整到与原始输入x相同的高和宽
#
#         value2 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(y, [height_y//6, width_y//6]))))
#         value2 = F.interpolate(value2, size=y.size()[2:], mode='bilinear', align_corners=True)
#
#         value3 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(y, [height_y//8, width_y//8]))))
#         value3 = F.interpolate(value3, size=y.size()[2:], mode='bilinear', align_corners=True)
#
#         value = torch.cat([value1, value2, value3], dim=1)
#         value = self.conv_1x1(value)
#         # 计算注意力分数
#         attention_scores = torch.matmul(query, key)
#         attention_scores = attention_scores / torch.sqrt(torch.tensor(self.out_channels//2, dtype=torch.float32))
#         # 计算注意力权重
#         attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
#         # 使用注意力权重对value进行加权平均
#         attended_value = torch.matmul(attention_weights, value)
#         # 使用注意力机制融合原始输入和加权平均后的值
#         output = x + self.gamma * attended_value
#         return output


class MASM(nn.Module):
    def __init__(self, in_channels, out_channels, height, p, d, a):
        super(MASM, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels*3, out_channels, kernel_size=1)
        num_dep = [2, 4, 4, 6, 9, 15, 0]  # 定义一个列表，用于多尺度自适应融合模块的深度参数
        rate = [x.item() for x in torch.linspace(0, 0.2, sum(num_dep))]  # 根据 num_dep 中的深度参数生成一个步长为0.2的列表，用于控制多尺度卷积的操作
        self.duo = MultiReceptiveFieldConvModule(in_channels, out_channels, rate[5], p, d, a)
        self.SH = AdaptiveAttention(in_channels, out_channels, height)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1, bias=False)  # 一维卷积

    def forward(self, x):
        x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))
        x2 = self.duo(x)
        x3 = self.SH(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv2(x)
        return x




# 定义transformer中的位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

        self.index = 0
        self.createdir = 0
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)

##改这里
            if name == 'layer21':
                #self.feature_vis(x)
                x1 = x
                #  合并每个特征图的热力图只保留一个
                num_channels, height, width = x1.shape[1], x1.shape[2], x1.shape[3]
                # 初始化一个与单个通道相同大小的零数组，用于存储合并后的特征图
                merged_feature_map = np.zeros((height, width))
                # 遍历所有通道，计算平均特征图
                for i in range(num_channels):
                    channel_data = x1[0, i].detach().to('cpu').numpy()
                    merged_feature_map += channel_data
                # 计算平均值
                merged_feature_map /= num_channels
                # 可视化合并后的特征图
                plt.figure(figsize=(8, 8))
                plt.imshow(merged_feature_map, cmap="viridis")

                plt.axis('off')
                plt.show()

                if not os.path.exists(
                    './crop_combine/combine512/CAM/feature_vis4_{:01}'.format(self.createdir)): os.makedirs(
                    './crop_combine/combine512/CAM/feature_vis4_{:01}'.format(self.createdir))

                plt.savefig('./crop_combine/combine512/CAM/feature_vis4_{:01}/{:01}.tif'.format(self.createdir, self.index),
                            dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()
                if self.index > 348:
                    self.index = 0
                    self.createdir += 1
                else:
                    self.index += 1
                # 可视化每个特征图的热力图
                #feature_map_data_list = [x1[0, i].detach().to('cpu').numpy() for i in range(x1.shape[1])]
                # plt.figure(figsize=(16, 16))
                # for i, feature_map_data in enumerate(feature_map_data_list):
                #     plt.subplot(8, 8, i + 1)
                #     plt.imshow(feature_map_data, cmap="viridis")
                #     plt.title(f"Feature Map {i + 1}")
                #     plt.axis('off')
                #     plt.show()




            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x


        return out

    #
    # def feature_vis(self,feats):  # feaats形状: [b,c,h,w]
    #     w = 512
    #     output_shape = (w, w)  # 输出形状
    #     channel_mean = torch.mean(feats, dim=1, keepdim=True)  # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
    #     #channel_mean,_ = torch.max(feats,dim=1,keepdim=True)
    #     #channel_mean = torch.var(feats, dim=1, keepdim=True)
    #     channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
    #     channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().detach().numpy()  # 四维压缩为二维
    #     channel_mean = (
    #             ((channel_mean - np.min(channel_mean)) / (np.max(channel_mean) - np.min(channel_mean))) * 255).astype(
    #         np.uint8)
    #     savedir = './crop_combine/combine512/CAM/'
    #     if not os.path.exists(savedir + 'feature_vis4_{:01}'.format(self.createdir )): os.makedirs(savedir + 'feature_vis4_{:01}'.format(self.createdir))
    #     channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
    #     filename = savedir + 'feature_vis4_{:01}/{:01}.tif'.format(self.createdir ,self.index)
    #     #print(filename)
    #     cv2.imwrite(filename, channel_mean)
    #     if self.index >28  :
    #         self.index = 0
    #         self.createdir +=1
    #     else:
    #         self.index += 1


class pbfam(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False , Regularization=None):
    # def __init__(self, in_channels, out_channels, height,num_classes, pretrain_backbone: bool = False):      #修改
        super(pbfam, self).__init__()
        backbone = resnet50()

        if pretrain_backbone:

            backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))
        self.regul = Regularization

        self.stage_out_channels = [64, 256, 512, 1024, 2048]
        return_layers = {'relu': 'out0', 'layer1': 'out1', 'layer2': 'out2', 'layer3': 'out3', 'layer4': 'out4'}

        return_layers1 = {'layer4': 'out4'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        # self.backbone_edge = IntermediateLayerGetter(backbone, return_layers=return_layers1)
        # self.new_conv_edeg = nn.Conv2d(1 ,3 , kernel_size=1, stride=1, padding=0 , bias=False )

        self.new_conv_edeg2 = nn.Conv2d(3, 256, kernel_size=1, stride=1, padding=0, bias=False)

        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3])
        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2])
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1])
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0])
        self.conv = OutConv(64, num_classes=num_classes)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.max_pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool8 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.SBA_1 = SBA(input_dim=256, output_dim=256)

        #self.new_conv_edeg3 = nn.Conv2d(256, 256, kernel_size=8, stride=2, padding=0, dilation=1, bias=False)
        #self.new_conv_edeg4 = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0, bias=False)

        self.new_conv_edeg5 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)

        #self.PSMF_ = PSFM(Channel=2048)
        self.SBA_2 = SBA(input_dim=256, output_dim=256)
        self.index = 0

        self.x_psmf = False
        self.createdir = 0
    #def forward(self, x: torch.Tensor , edge: torch.Tensor) -> Dict[str, torch.Tensor]:
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = x
        x_psmf =x
        input_shape = x.shape[-2:]
        result = OrderedDict()

        backbone_out = self.backbone(x)
        x = self.up1(backbone_out['out4'], backbone_out['out3'])  # transformer修改

        x = self.up2(x, backbone_out['out2'])  # [16,512,16,16]

        x = self.up3(x, backbone_out['out1'])  # [16,256,32,32]

        x_pool = self.new_conv_edeg2(x1)
        x_pool1 = self.max_pool1(x_pool)
        x_pool3 = self.max_pool3(x_pool)
        x_pool6 = self.max_pool6(x_pool)
        x_pool8 = self.max_pool8(x_pool)
        x_sba1 = self.SBA_1(x_pool1, x_pool6)
       # x_sba2 = self.SBA_2(x_pool3, x_pool8)

        # x_sba1 = self.new_conv_edeg3(x_sba1)
        # x_sba2 = self.new_conv_edeg4(x_sba2)
        if self.x_psmf:
            x_cat_sba_psmf = F.normalize(torch.cat((self.regul* x_sba1, x_psmf ), dim=1) , p=2, dim=1)
        else:
            x_psmf = self.up4(x, backbone_out['out0'])  # [16,64,64,64]

        #  x = self.new_conv_edeg5(F.normalize(torch.cat((0.001 * x_sba1), dim=1), p=2, dim=1)) + x_psmf
        x = self.new_conv_edeg5(F.normalize(self.regul* x_sba1, dim=1)) + x_psmf

        x = self.conv(x)
        # self.feature_vis(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)


        result["out"] = x
        return result

    # def feature_vis(self,feats):  # feaats形状: [b,c,h,w]
    #     w = 512
    #     output_shape = (w, w)  # 输出形状
    #     channel_mean = torch.mean(feats, dim=1, keepdim=True)  # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
    #     #channel_mean,_ = torch.max(feats,dim=1,keepdim=True)
    #     #channel_mean = torch.var(feats, dim=1, keepdim=True)
    #     channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
    #     channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().detach().numpy()  # 四维压缩为二维
    #     channel_mean = (
    #             ((channel_mean - np.min(channel_mean)) / (np.max(channel_mean) - np.min(channel_mean))) * 255).astype(
    #         np.uint8)
    #     savedir = './crop_combine/combine512/CAM/'
    #     if not os.path.exists(savedir + 'feature_vis4_{:01}'.format(self.createdir )): os.makedirs(savedir + 'feature_vis4_{:01}'.format(self.createdir))
    #     channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
    #     filename = savedir + 'feature_vis4_{:01}/{:01}.tif'.format(self.createdir ,self.index)
    #     #print(filename)
    #     cv2.imwrite(filename, channel_mean)
    #     if self.index >298  :
    #         self.index = 0
    #         self.createdir +=1
    #     else:
    #         self.index += 1

