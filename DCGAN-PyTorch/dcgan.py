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

""""
# Define the Generator Network
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(params['nz'], params['ngf']*8,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(params['ngf']*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(params['ngf']*8, params['ngf']*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ngf']*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(params['ngf']*4, params['ngf']*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ngf']*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(params['ngf']*2, params['ngf'],
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ngf'])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(params['ngf'], params['nc'],
            4, 2, 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = F.tanh(self.tconv5(x))

        return x
"""

# Generator Code

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
    


"""
# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 128 x 128
        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
            kernel_size=4, stride=2, padding=1, bias=False)  # Salida: (ndf) x 64 x 64

        # Input Dimension: (ndf) x 64 x 64
        self.conv2 = nn.Conv2d(params['ndf'], params['ndf'] * 2,
            kernel_size=4, stride=2, padding=1, bias=False)  # Salida: (ndf*2) x 32 x 32
        self.bn2 = nn.BatchNorm2d(params['ndf'] * 2)

        # Input Dimension: (ndf*2) x 32 x 32
        self.conv3 = nn.Conv2d(params['ndf'] * 2, params['ndf'] * 4,
            kernel_size=4, stride=2, padding=1, bias=False)  # Salida: (ndf*4) x 16 x 16
        self.bn3 = nn.BatchNorm2d(params['ndf'] * 4)

        # Input Dimension: (ndf*4) x 16 x 16
        self.conv4 = nn.Conv2d(params['ndf'] * 4, params['ndf'] * 8,
            kernel_size=4, stride=2, padding=1, bias=False)  # Salida: (ndf*8) x 8 x 8

        # Input Dimension: (ndf*8) x 8 x 8
        self.conv5 = nn.Conv2d(params['ndf'] * 8, 1,
            kernel_size=4, stride=1, padding=0, bias=False)  # Salida: 1 x 5 x 5

        # Capa final para obtener una salida de tamaño (b_size)
        self.fc = nn.Linear(5 * 5, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)  # Salida: (ndf) x 64 x 64
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)  # Salida: (ndf*2) x 32 x 32
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)  # Salida: (ndf*4) x 16 x 16
        x = F.leaky_relu(self.conv4(x), 0.2, True)  # Salida: (ndf*8) x 8 x 8

        x = self.conv5(x)  # Salida: 1 x 5 x 5
        x = x.view(-1, 5 * 5)  # Reestructurando a (b_size, 25)
        x = torch.sigmoid(self.fc(x))  # Salida: (b_size, 1)
        # x = x.view(-1, 1)

        return x.view(-1)  # Asegurando que la salida es de tamaño (b_size)

"""

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