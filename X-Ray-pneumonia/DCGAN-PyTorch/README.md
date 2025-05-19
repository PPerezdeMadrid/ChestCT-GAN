# Appendix: Generating Chest X-ray Images with Pneumonia

This project uses a **DCGAN (Deep Convolutional Generative Adversarial Network)** model to generate synthetic chest X-ray images showing pneumonia. Below are the steps to train the model and generate images.

## Project Structure

```bash
DCGAN-PyTorch/
├── README.md                
├── config.json              # Configuration file (paths, parameters)
├── dcgan512.py              # DCGAN implementation (512x512 resolution)
├── dcgan.py                 # Base DCGAN model (64x64 resolution)
├── eval_model.py            # Script to evaluate the model
├── generate.py              # Script to generate images
├── graphLogs.py             # Functions to plot training logs
├── images_prueba/           # Folder for generated test images
├── model_prueba/            # Folder for saved models
├── requirements_xray.txt    # Project dependencies (installation)
├── train.py                 # Script to train the model
├── main.py                  # Main script to launch training
├── X-Ray_TC_dcgan.gif       # Animation of training progress
└── utils.py                 # Utility functions
```

## Prerequisites

1. Python 3.6+ must be installed.
2. Install the required dependencies. Run the following command:

```bash
pip install -r requirements_xray.txt
```

## Steps to Train the Model

1. **Download Data from Kaggle**
   If the dataset has not been downloaded yet, uncomment the following lines in `main.py`:

   ```python
   # Download and prepare the data
   # data_path = download_xray_data()
   # prepare_data(data_path,"../Data_train")
   ```

   This will automatically download the chest X-ray dataset from Kaggle and prepare it for training.

2. **Run the Training Process**
   Once the data is ready, you can train the model using the command:

   ```bash
   python main.py
   ```

   The model will start training, and results will be saved in the directories defined in `config.json`.

## Key Files and Directories

* **Saved models**: Trained models will be stored in the `model_prueba/` folder.
* **Generated images**: Images generated during training will be stored in the `images_prueba/` folder.
* **Evaluation**: Model evaluation results will be saved in the `evaluation/` folder.
* **Logs and plots**: Training logs and plots can be visualized using the functions in `graphLogs.py`.

## Configuration

The configuration file `config.json` defines key parameters such as data paths, model architecture, and training options. Be sure to review and modify it according to your needs.

## Generate 512x512 Images

If you want to generate 512x512 images instead of the default resolution, follow these steps:

1. **Update Resolution in `config.json`**
   Open `config.json` and set `"imsize"` to `512`. It’s also recommended to set the batch size (`bsize`) to `32` for better performance. The relevant section should look like this:

   ```json
   {
      "params": {
         "bsize": 128,
         "imsize": 64, 
         "nc": 1,
         "nz": 100,
         "ngf": 128,
         "ndf": 128,
         "nepochs": 1000,
         "lr": 0.0001,
         "beta1": 0.5,
         "beta2": 0.999,
         "save_epoch": 100
      },
      "model": {
         "path_dcgan": "model_prueba/model_dcgan",
         "path_wgan": "model_prueba/model_wgan",
         "image_path_dcgan": "images_prueba/images_dcgan",
         "image_path_wgan": "images_prueba/images_wgan",
         "evaluation_dcgan": "evaluation_prueba/evaluation_dcgan",
         "evaluation_wgan": "evaluation_prueba/evaluation_wgan"
      },
      "datasets":{
         "chestKaggle": "../../../../TFG/ChestCTKaggle/Data/",
         "nbia": "../../src/Pipeline/Data/Data-Transformed",
         "xray": "../Data_train/"
      }
   }
   ```

2. **Switch the Training Script**
   In `train.py`, modify the import statement:

   From:

   ```python
   from dcgan import Generator, Discriminator, weights_init
   ```

   To:

   ```python
   from dcgan512 import Generator, Discriminator, weights_init
   ```

   This ensures the training uses the version of the model built for 512x512 image generation.

---

## Scripts

### `generate.py`

This script generates synthetic images from a pre-trained model.

#### Parameters:

* **`-load_path`**:
  Path to the checkpoint (pre-trained model) to load. By default, this is set to `f'{model_path}/{model_pth}'`, which uses the values defined in the config file.

  * **Type**: string
  * **Description**: Specifies the model file to be loaded for image generation.

* **`-num_output`**:
  Number of images to generate. The default is 5.

  * **Type**: integer
  * **Description**: Defines how many images to generate.

#### Example Usage:

```bash
python generate.py -load_path 'path/to/model.pth' -num_output 5
```

---

### `eval_model.py`

This script is used to evaluate the performance of a trained model.

#### Parameters:

* **`model_name`**:
  The name of the model file to evaluate. Default is `"model_epoch_400.pth"`.

  * **Type**: string
  * **Description**: The specific model file to evaluate.

* **`model_path`**:
  The path where the model is stored. This is built from the value in the config file: `config["model"]["path_dcgan"] + model_name`.

  * **Type**: string
  * **Description**: Full path to the model to evaluate.

#### Example Usage:

```bash
python eval_model.py
```

This command will evaluate the model using the default path. If you want to evaluate a different model, modify `model_name` in the code or add an argument to the script.

---

### Notes

* Classifier model that can be improved: [https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia)
* Dataset: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download)

