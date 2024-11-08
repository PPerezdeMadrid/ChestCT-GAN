import torch, json
from utils import get_chestct 
from dcgan import Generator, Discriminator


def print_green(text):
    print("\033[92m" + text + "\033[0m")

print_green("============== Starting Evaluation ==============")

# leer parámetros y modelo:
with open('config.json', 'r') as json_file:
    config = json.load(json_file)

model_path = config["model"]["path"]

print_green("Parameters uploaded")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_green(f'{device} will be used.\n')

checkpoint = torch.load(f'{model_path}/model_ChestCT.pth', weights_only=True)
params = checkpoint['params']

# Get the data
dataloader = get_chestct(params)

# Load the generator and discriminator
netG = Generator(params).to(device)
netD = Discriminator(params).to(device)

# Cargar el modelo entrenado con weights_only=True para una mayor seguridad
# checkpoint = torch.load(f'{model_path}/model_ChestCT.pth', weights_only=True)
checkpoint = torch.load(f'{model_path}/model_ChestCT.pth', map_location=device, weights_only=True)
netG.load_state_dict(checkpoint['generator'])
netD.load_state_dict(checkpoint['discriminator'])

print_green("Está todo biennn")

