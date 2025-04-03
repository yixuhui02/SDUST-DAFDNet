########################################################################################################################
# 画loss图
########################################################################################################################

import os
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt


class LossHistory():
    def __init__(self, save_path , val_flag=True):
        self.save_path = save_path
        self.val_flag = val_flag

        self.losses = []
        self.acc = []
        if self.val_flag:
            self.val_loss = []
            self.val_acc = []
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def append_loss(self, loss , loss_test):
        self.losses.append(loss)
        self.val_loss.append(loss_test)
        self.loss_plot()

    def append_acc(self, acc ,acc_test):
        self.acc.append(acc)
        self.val_acc.append(acc_test)
        self.acc_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()

        plt.plot(iters, self.losses, 'red', linewidth=2, label='Train loss')


        if self.val_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='Test loss')


        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_"  + ".png"))

        plt.cla()
        plt.close("all")

    def acc_plot(self):
        iters = range(len(self.acc))

        plt.figure()

        plt.plot(iters, self.acc, 'red', linewidth=2, label='Train acc')


        if self.val_flag:
            plt.plot(iters, self.val_acc, 'coral', linewidth=2, label='Test acc')


        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_acc_" + ".png"))

        plt.cla()
        plt.close("all")

