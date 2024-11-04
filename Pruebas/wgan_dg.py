# -*- coding: utf-8 -*-
"""WGAN-DG.ipynb

pip install numpy<2
pip install --upgrade --force-reinstall tensorflow
Win + R >> regedit >> HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem >> LongPathsEnabled = 1
pip install --upgrade numpy


Original file is located at
    https://colab.research.google.com/drive/1-VBIAnnxrI1DZVJK6nAjfbObVfSPYq71

Chest CT Data generation with a WGAN
"""

import kagglehub

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from itertools import chain
from torchvision import utils
import tensorflow as tf

"""# Dataloader"""

# path= "chest-ctscan-images/versions/1/Data/"
path = "../../../ChestCTKaggle/Data"

def get_data_loader(args):
      train_path = path+"train"
      valid_path = path+"valid"
      test_path = path+"test"

      transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64,64)),
        transforms.ToTensor(),  # Convertir a tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalización a [-1, 1]
      ])
        # Cargar los datasets
      train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
      valid_dataset = datasets.ImageFolder(root=valid_path, transform=transform)
      test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

      # Verificar el tamaño del dataset
      print(f"===> Tamaño del conjunto de entrenamiento: {len(train_dataset)}")
      print(f"===> Tamaño del conjunto de validación: {len(valid_dataset)}")
      print(f"===> Tamaño del conjunto de prueba: {len(test_dataset)}")

      train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
      test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

      return train_dataloader, test_dataloader

"""# Generator and Discriminator"""

import WGAN as wgan



import argparse
import os

"""
python main.py --model WGAN-GP --is_train True --download True --dataset chestCT --generator_iters 40000 --batch_size 64
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of GAN models.")

    parser.add_argument('--model', type=str, default='WGAN-GP', choices=['GAN', 'DCGAN', 'WGAN-CP', 'WGAN-GP'])
    parser.add_argument('--is_train', type=str, default='True')
    parser.add_argument('--dataroot', required=False, help='path to dataset')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'chestCT'],
                            help='The name of dataset')
    parser.add_argument('--download', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--cuda',  type=str, default='False', help='Availability of cuda')

    parser.add_argument('--load_D', type=str, default='False', help='Path for loading Discriminator network')
    parser.add_argument('--load_G', type=str, default='False', help='Path for loading Generator network')
    parser.add_argument('--generator_iters', type=int, default=10000, help='The number of iterations for generator in WGAN model.')
    return check_args(parser.parse_args())


# Checking arguments
def check_args(args):
    # --epoch
    try:
        assert args.epochs >= 1
    except:
        print('Number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larger than or equal to one')

    if args.dataset == 'cifar' or args.dataset == 'stl10':
        args.channels = 3
    else:
        args.channels = 1
    args.cuda = True if args.cuda == 'True' else False
    return args

# parse_args
# get_data_loader

def main(args):
    model = None
    if args.model == 'GAN':
        print("Modelo GAN no creado")
    elif args.model == 'DCGAN':
        print("Modelo DCGAN no creado")
    elif args.model == 'WGAN-CP':
        print("Modelo WGAN-CP no creado")
    elif args.model == 'WGAN-GP':
        print("Has elegido WGAN-GP")
        model = wgan.WGAN_GP(args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)

    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(args)
    #feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

    # Start model training
    if args.is_train == 'True':
        model.train(train_loader)

    # start evaluating on test data
    else:
        model.evaluate(test_loader, args.load_D, args.load_G)
        for i in range(50):
           model.generate_latent_walk(i)

class Args:
    def __init__(self):
        self.model = 'WGAN-GP'  # Cambia según el modelo que deseas usar
        self.is_train = 'True'   # Cambia a 'False' si solo quieres evaluar
        self.batch_size = 32      # Ajusta según tu configuración
        self.cuda = True           # Cambia según si usas CUDA
        self.load_D = None        # Ruta para cargar el modelo del discriminador
        self.load_G = None
        self.channels = 1            # Número de canales (por ejemplo, 3 para imágenes RGB)
        self.generator_iters = 10

args = Args()

main(args)