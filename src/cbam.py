import torch
from torch import nn

class channel_attention(nn.Module):
    def __init__(self,channel , ratio=16 ):
        super(channel_attention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel//ratio ,False),
            nn.ReLU(),
            nn.Linear(channel//ratio , channel ,False),

        )
        self.sigmoid = nn.Sigmoid()

    def forward(self ,x):
        b,c,h ,w = x.size()
        max_pool = self.max_pool(x).view([b,c])
        avg_pool = self.avg_pool(x).view([b,c])

        max_pool = self.fc(max_pool)
        avg_pool = self.fc(avg_pool)

        out = max_pool+avg_pool
        out = self.sigmoid(out).view([b,c,1,1])

        return out*x


class spacial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spacial_attention, self).__init__()

        padding = kernel_size//2
        self.conv = nn.Conv2d(2,1,kernel_size ,1, padding , bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool ,_= torch.max(x,dim =1 ,keepdim=True)
        mean_pool = torch.mean(x , dim=1 , keepdim=True)
        pool_out = torch.cat([max_pool,mean_pool],dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)

        return out * x

class Cbam(nn.Module):
    def __init__(self,channel , ratio = 16 , kernel_size=7):
        super(Cbam, self).__init__()
        self.channel = channel_attention(channel , ratio = 16)
        self.spacial = spacial_attention(kernel_size)

    def forward(self ,x ):
        x = self.channel(x)
        x = self.spacial(x)
        return x


model = Cbam(512)
inputs = torch.ones([8,512,26,26])

outputs = model(inputs)

print(outputs.shape)
