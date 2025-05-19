import train
import json
from utils import download_xray_data, prepare_data

def load_config():
    with open('config.json', 'r') as json_file:
        return json.load(json_file)

print("===> Training:")

# Download and prepare the data
# data_path = download_xray_data()
# prepare_data(data_path,"../Data_train")

config = load_config()
train.main(model_type="dcgan", params=config["params"])