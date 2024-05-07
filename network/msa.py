import torch.nn as nn
from torch.nn.parameter import Parameter
import torch

class Stylemodule(nn.Module):
    def __init__(self, channel):
        super(Stylemodule, self).__init__()
        self.cfc = Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)
        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()
    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1
        z_hat = self.bn(z)
        g = self.activation(z_hat)
        return g
    def forward(self, x):
        t = self._style_pooling(x)
        g = self._style_integration(t)
        return x * g

class Scalemodule(nn.Module):
    def __init__(self, nin):
        super(Scalemodule, self).__init__()
        self.conv_dws = nn.Conv2d(
            nin, nin, kernel_size=1, stride=1, padding=0, groups=nin
        )
        self.bn_dws = nn.BatchNorm2d(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_point = nn.Conv2d(
            nin, 1, kernel_size=1, stride=1, padding=0, groups=1
        )
        self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        out = self.conv_dws(x)
        out = self.bn_dws(out)
        out = self.relu_dws(out)
        out = self.maxpool(out)
        out = self.conv_point(out)
        out = self.bn_point(out)
        out = self.relu_point(out)
        m, n, p, q = out.shape
        out = self.softmax(out.view(m, n, -1))
        out = out.view(m, n, p, q)
        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        # 将输入与输出相乘并加到原输入上
        out = torch.mul(out, x)
        out = out + x
        return out
class msa(nn.Module):
    def __init__(self, channel):
        super(msa, self).__init__()
        self.style_layer = Stylemodule(channel)
        self.scale_layer = Scalemodule(channel)

    def forward(self, x):
        style_output = self.style_layer(x)
        scale_output = self.scale_layer(x)
        fused_output = style_output + scale_output
        return fused_output


