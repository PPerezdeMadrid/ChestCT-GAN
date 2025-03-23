import train
import json

def load_config():
    with open('config.json', 'r') as json_file:
        return json.load(json_file)

print("===> Entrenamiento:")
config = load_config()
train.main(model_type="dcgan", params=config["params"])