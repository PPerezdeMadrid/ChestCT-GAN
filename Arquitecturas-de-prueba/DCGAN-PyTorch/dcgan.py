import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)




# Generator Code
"""
class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        # self.ngpu = ngpu --> lo añade como param 
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( params['nz'],params['ngf'] * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(params['ngf'] * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(params['ngf'] * 8, params['ngf'] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params['ngf'] * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( params['ngf'] * 4, params['ngf'] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params['ngf'] * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( params['ngf'] * 2, params['ngf'], 4, 2, 1, bias=False),
            nn.BatchNorm2d(params['ngf']),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( params['ngf'], params['nc'], 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)



class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(params['nc'], params['ndf'], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(params['ndf'], params['ndf'] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params['ndf'] * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(params['ndf'] * 2, params['ndf'] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params['ndf'] * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(params['ndf'] * 4, params['ndf'] * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(params['ndf'] * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(params['ndf'] * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
"""

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(params['nz'], params['ngf'] * 16, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(params['ngf'] * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(params['ngf'] * 16, params['ngf'] * 8, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(params['ngf'] * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(params['ngf'] * 8, params['ngf'] * 4, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(params['ngf'] * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(params['ngf'] * 4, params['ngf'] * 2, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(params['ngf'] * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(params['ngf'] * 2, params['ngf'], 4, 2, 1, bias=False),  # 64x64
            nn.BatchNorm2d(params['ngf']),
            nn.ReLU(True),

            nn.ConvTranspose2d(params['ngf'], params['ngf'] // 2, 4, 2, 1, bias=False),  # 128x128
            nn.BatchNorm2d(params['ngf'] // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(params['ngf'] // 2, params['ngf'] // 4, 4, 2, 1, bias=False),  # 256x256
            nn.BatchNorm2d(params['ngf'] // 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(params['ngf'] // 4, params['nc'], 4, 2, 1, bias=False),  # 512x512
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(params['nc'], params['ndf'] // 4, 4, 2, 1, bias=False),  # 256x256
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(params['ndf'] // 4, params['ndf'] // 2, 4, 2, 1, bias=False),  # 128x128
            nn.BatchNorm2d(params['ndf'] // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(params['ndf'] // 2, params['ndf'], 4, 2, 1, bias=False),  # 64x64
            nn.BatchNorm2d(params['ndf']),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(params['ndf'], params['ndf'] * 2, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(params['ndf'] * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(params['ndf'] * 2, params['ndf'] * 4, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(params['ndf'] * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(params['ndf'] * 4, params['ndf'] * 8, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(params['ndf'] * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(params['ndf'] * 8, params['ndf'] * 16, 4, 2, 1, bias=False),  # 4x4
            nn.BatchNorm2d(params['ndf'] * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(params['ndf'] * 16, 1, 4, 1, 0, bias=False),  # 1x1
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
