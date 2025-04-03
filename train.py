########################################################################################################################
# 训练脚本
########################################################################################################################

import os
import time
import datetime

import torch
import numpy as np
from src import pbfam
from src import bsmsm
from train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import VOCSegmentation
from callbacks import LossHistory
import transforms as T
import warnings

from src.deeplabv3_plus import DeepLab
from src.unet_model import UNet                                                  #unet修改1

warnings.filterwarnings("ignore")

class SegmentationPreset:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        w = 128
        trans = [T.Resize(w, w)]                                                                        # 尺寸修改
        trans.append(T.ToTensor())
        trans.append(T.Normalize(mean=mean, std=std))
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_transform():
    return SegmentationPreset()

def create_model(num_classes, pretrain=True , Regularization=None):
    # model = unet_resnet50(num_classes=num_classes, pretrain_backbone=True)
    model = pbfam(num_classes=num_classes, pretrain_backbone=True ,Regularization = Regularization)
    # weights_path = './crop_combine/combine600/iou91/iou91.pth'  # 权重路径
    # assert os.path.exists(weights_path), f"weights {weights_path} not found."
    # weights_dict = torch.load(weights_path, map_location='cpu')
    # model.load_state_dict(weights_dict)
    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes

    results_file = './logs/results{}.txt'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    loss_history = LossHistory("./logs/")

    train_dataset = VOCSegmentation(args.data_path,
                                    transforms__=get_transform(), txt_name="train.txt")

    val_dataset = VOCSegmentation(args.data_path,
                                  transforms__=get_transform(), txt_name="val.txt")

    num_workers = 2                                                                                                     # 加载数据使用cpu线程数

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=False,
                                               #collate_fn=train_dataset.collate_fn,
                                               drop_last = True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=False,
                                             #collate_fn=val_dataset.collate_fn,
                                             drop_last = True)

    model = create_model(num_classes=num_classes, pretrain=False , Regularization=  args.Regularization)    #修改
    #model  = DeepLab(num_classes=num_classes, backbone="mobilenet", downsample_factor=16, pretrained=False)
    #model = UNet(n_channels=3,n_classes=num_classes)                                # unet修改2

    # model = HighResolutionNet(base_channel=32, num_joints=num_classes)               #HRnet
    model.to(device)

    params_to_optimize = [
        {"params": [p for p in model.parameters() if p.requires_grad]}
    ]

    #optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    start_time = time.time()
    train_MAX_MIOU = 0
    eval_MAX_MIOU = 0
    for epoch in range(0, args.epochs):
        train_mean_loss, lr ,train_info_confmat = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler , num_classes=num_classes)

        train_info_= str(train_info_confmat)
        print(train_info_)
        val_mean_loss, confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        #
        # if train_info_confmat.get_iou() >= train_MAX_MIOU:
        #     train_MAX_MIOU = train_info_confmat.get_iou()
        #     torch.save(model.state_dict(), 'save_weights/best_train_iou_finally_module.pth')            # 修改3
        #
        # if confmat.get_iou() >= eval_MAX_MIOU:
        #     eval_MAX_MIOU = confmat.get_iou()
        #     torch.save(model.state_dict(), 'save_weights/best_eval_iou_finally_module.pth')         # 修改4
        # # write into txt
        # with open(results_file, "a") as f:
        #     train_info = f"[epoch: {epoch}]\n" \
        #                  f"train_loss: {train_mean_loss:.4f}\n" \
        #                  f"val_loss: {val_mean_loss:.4f}\n" \
        #                  f"lr: {lr:.6f}\n"
        #     f.write(train_info + train_info_ +val_info + "\n\n")
        #
        #
        # loss_history.append_loss(train_mean_loss, val_mean_loss)
        # loss_history.append_acc(train_info_confmat.get_acc(), confmat.get_acc())

        # torch.save(model.state_dict(), 'save_weights/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch, train_mean_loss, val_mean_loss))
        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'save_weights/128-ep%03d-loss%.3f-iou%.3f.pth' % (epoch, train_mean_loss,train_info_confmat.get_iou()))

        #torch.save(model.state_dict(), 'save_weights/keshihua/128-ep%03d-loss%.3f-iou%.3f.pth' % (
        #epoch, train_mean_loss, train_info_confmat.get_iou()))

        # torch.save(model.state_dict(),
        #            'save_weights/model.pth' )
        # torch.save(model.state_dict(),
        #            'save_weights/model_finally_module.pth')                            # 修改5
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
    print("model save---->",'save_weights')
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch training")
    parser.add_argument("--data-path", default='./crop_combine/train', help="VOCdevkit root")  # 数据集路径
    parser.add_argument("--num-classes", default=5, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=32, type=int)                                                      # batch_size
    parser.add_argument("--epochs", default=30, type=int, metavar="N",                                                  # epochs
                        help="number of total epochs to train")
    parser.add_argument("--Regularization", default=0.001, type=int, metavar="Regularization",                                                  # epochs
                        help="Regularization train")
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')                                    # 打印频率
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cud"
                             "a.amp for mixed precision training")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
