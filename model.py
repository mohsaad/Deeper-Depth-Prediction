# Mohammad Saad
# Model for Deeper Depth Prediction
# 2/19/2018

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, d1, d2, stride = 1):
        super(ResidualBlock, self).__init__()

        # leading into d1
        self.conv1 = nn.Conv2d(in_channels, d1, 1, stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(d1)
        self.relu1 = nn.ReLU(inplace = True)

        # leading into d1-2
        self.conv2 = nn.Conv2d(d1, d1, 3, padding = True, bias = False)
        self.bn2 = nn.BatchNorm2d(d1)
        self.relu2 = nn.ReLU(inplace = True)

        # leading into d2
        self.conv3 = nn.Conv2d(d1, d2, 1, bias = False)
        self.bn3 = nn.BatchNorm2d(d2)

        # if not self.skip:
        #     self.conv4 = nn.Conv2d(in_channels, d2, stride, bias = False)
        #     self.bn4 = nn.BatchNorm2d(d2)

        # final Relu at end of layer
        self.relu3 = nn.ReLU(inplace = True)


    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x

        out = self.relu3(out)

        return out

class ProjectionBlock(nn.Module):

    def __init__(self, in_channels, d1, d2, stride = 1):
        super(ProjectionBlock, self).__init__()

        # feeding into first d1 block
        self.conv1 = nn.Conv2d(in_channels, d1, 1, stride = 5, bias = False)
        self.bn1 = nn.BatchNorm2d(d1)
        self.relu1 = nn.ReLU(inplace = True)

        # feeding into second d1 block
        self.conv2 = nn.Conv2d(d1, d1, 3, padding = 2, bias = False)
        self.bn2 = nn.BatchNorm2d(d1)
        self.relu2 = nn.ReLU(inplace = True)

        # feeding into first d2 block
        self.conv3 = nn.Conv2d(d1, d2, 1, bias = False)
        self.bn3 = nn.BatchNorm2d(d2)

        # feeding into second d2 block
        self.conv4 = nn.Conv2d(in_channels, d2, 1, stride = 5, bias = False)
        self.bn4 = nn.BatchNorm2d(d2)

        self.relu3 = nn.ReLU(inplace = True)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # do residual branch
        residual = x
        residual = self.conv4(residual)
        residual = self.bn4(residual)

        out += residual

        out = self.relu3(out)

        return out

# Fast Up Convolution from the paper, including the interleaving step

class FastUpConvolution:

    def __init__(self, in_channels, out_channels, batch_size):
        super(FastUpConvolution, self).__init__()

        self.batch_size = batch_size

        # do 4 convolutions on the same output with different kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3,3))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (2,3))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (3,2))
        self.conv4 = nn.Conv2d(in_channels, out_channels, (2,2))

    # interleaving operation
    def interleave(self, out1, out2, out3, out4):



        return

    def forward(self, x):
        out1 = self.conv1(x, nn.functional.pad(x, (1,1,1,1)))
        out2 = self.conv2(x, nn.functional.pad(x, (1,1,1,0)))
        out3 = self.conv3(x, nn.functional.pad(x, (1,0,1,1)))
        out4 = self.conv4(x, nn.functional.pad(x, (1,0,1,0)))

        out = self.interleave(out1, out2, out3, out4)

        return out

class FastUpProjection:

    def __init__(self, in_channels, out_channels, batch_size):
        super(FastUpProjection, self).__init__()

        self.UpConv1 = FastUpConvolution(in_channels, out_channels, batch_size)
        self.relu1 = nn.ReLU(inplace = True)

        self.UpConv2 = FastUpConvolution(in_channels, out_channels, batch_size)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.relu2 = nn.ReLU(inplace = True)

    def forward(self, x):
        out1 = self.UpConv1.forward(x)
        out2 = self.UpConv2.forward(x)

        out1 = self.relu1(out1)
        out1 = self.conv1(out1)

        out = out1 + out2
        out = self.relu2(out)

        return out
