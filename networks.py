import torch
import torch.nn as nn
import torch.nn.functional as F

class generator1(nn.Module):
    # initializers
    def __init__(self, d=128, r=16):
        super(generator1, self).__init__()
        self.Econv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.Econv1_bn = nn.BatchNorm2d(d)
        self.Econv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.Econv2_bn = nn.BatchNorm2d(d*2)
        self.Econv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.Econv3_bn = nn.BatchNorm2d(d*4)
        self.Econv4 = nn.Conv2d(d * 4, d, 4, 1, 0)
        self.Econv4_bn = nn.BatchNorm2d(d)
        self.Econv5 = nn.Conv2d(d, r, 1, 1, 0)
        
        self.deconv0 = nn.ConvTranspose2d(r, d, 1, 1, 0)
        self.deconv0_bn = nn.BatchNorm2d(d)
        self.deconv1 = nn.ConvTranspose2d(d, d*4, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.relu(self.Econv1_bn(self.Econv1(input)))
        x = F.relu(self.Econv2_bn(self.Econv2(x)))
        x = F.relu(self.Econv3_bn(self.Econv3(x)))
        x = F.relu(self.Econv4_bn(self.Econv4(x)))
        x = F.tanh(self.Econv5(x))
        v = x + (torch.round((x.data+1)/2)*2-1-x.data)
        
        x = F.relu(self.deconv0_bn(self.deconv0(v)))
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = self.deconv4(x)

        return x, v

class generator2(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator2, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*4, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = self.deconv4(x)

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=16, rate=16):
        super(discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(rate, d * 8, 1, 1, 0)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv4 = nn.Conv2d(d * 4, d*8, 4, 1, 0)
        self.conv5 = nn.Conv2d(d * 16, d * 4, 1, 1, 0)
        self.conv6 = nn.Conv2d(d * 4, 1, 1, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input1, input2):
        x = F.leaky_relu(self.conv1_1(input1), 0.2)

        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        y = F.leaky_relu(self.conv1_2(input2), 0.2)
        x = torch.cat([x, y], 1)

        x = F.leaky_relu(self.conv5(x), 0.2)
        x = self.conv6(x)

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
