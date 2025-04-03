import os
import sys
import torch
import numpy as np
from skimage import io
import torch.nn.functional as F
from torch.utils.data import DataLoader



class_name = ["building", "Push pile earth", "Pond water surface",
              "highway", "railway", "Park green space", "background"]
class_pixel_value = [200, 150, 100, 250, 220, 50, 0]
class_label_value = [4, 3, 2, 6, 5, 1, 0]

input_path = sys.argv[1]
output_path = sys.argv[2]

os.makedirs(output_path, exist_ok=True)


def label_to_visualization_map(image_array):
    colormap = np.sort(np.asarray(class_pixel_value, dtype='uint8'))
    return colormap[image_array]


net = Net(channels=3, num_classes=7).cuda()
net.load_state_dict(torch.load("./trained_model/DBCFNet.pth"))
net.eval()

test_dataset = dp.DataTest(input_path)
test_loader = DataLoader(test_dataset, batch_size=1)

for vi, data in enumerate(test_loader):
    imgs_A, imgs_B = data
    imgs_A = imgs_A.cuda().float()
    imgs_B = imgs_B.cuda().float()

    with torch.no_grad():
        out_change, outputs_B = net(imgs_A, imgs_B)
        out_change = F.sigmoid(out_change)
        outputs_B = outputs_B.cpu().detach()
        change_mask = out_change.cpu().detach() > 0.5
        change_mask = change_mask.squeeze()
        pred_B = torch.argmax(outputs_B, dim=1).squeeze()
        pred_B = (pred_B * change_mask.long()).numpy()

        io.imsave(os.path.join(output_path, test_dataset.get_mask_name(vi)), label_to_visualization_map(pred_B))
