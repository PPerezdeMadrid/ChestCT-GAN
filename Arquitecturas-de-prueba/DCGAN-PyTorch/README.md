# DCGAN

This project implements a **DCGAN (Deep Convolutional Generative Adversarial Network)** in PyTorch to generate synthetic images from medical datasets, specifically chest computed tomography (CT) scans.

## 📌 What is a DCGAN?

A **DCGAN** is a variant of generative adversarial networks (GANs) that uses deep convolutional layers. GANs consist of two networks competing against each other:

- **Generator:** generates fake images that mimic real ones.
- **Discriminator:** tries to distinguish between real and fake images.

During training, both models improve simultaneously: the generator learns to fool the discriminator, and the discriminator becomes better at differentiating. In the case of a DCGAN, deep convolutional architectures are used to better capture the visual features of images.

This approach has become especially useful in medicine, where the **lack of large volumes of labeled data** limits the training of robust models. By generating realistic synthetic images, datasets can be enriched and diagnostic algorithms improved.

---

## 🗂 Project Structure

```
DCGAN-PyTorch/
├── ChestTC_dcgan_*.gif           # Training GIF every certain epochs (only generated when trained)
├── config.json                   # Base configuration for training
├── dcgan.py                     # Main DCGAN model architecture
├── dcgan512.py                  # Variant for 512x512 images
├── train.py                     # Script to train the model
├── generate.py                  # Script to generate new images
├── eval_model.py                # Evaluation of the trained model
├── graphLogs.py                 # Visualization of training metrics
├── requirements_dcgan.txt       # Required dependencies
├── README.md                    
├── 1stHiperparams.json          # First hyperparameters file 
├── 2ndHiperparams.json          # Second hyperparameters file
├── 3rdHiperparams.json          # Third hyperparameters file
├── FinalConfig.json             # Final configuration
└── InitialConfig.json           # Initial configuration
```

---


## 🚀 Train the model

The `train.py` script is used to train  a **DCGAN**  model on medical image datasets.


```bash
python train.py -h
```

Output:

```
usage: train.py [-h] [--model {dcgan,wgan}] [--dataset {chestct,nbia}] [--configFile CONFIGFILE]

Train a DCGAN or WGAN model.

options:
  -h, --help            show this help message and exit
  --model {dcgan,wgan}  Choose between "dcgan" and "wgan" models to train.
  --dataset {chestct,nbia}
                        Choose the dataset: "chestct" or "nbia".
  --configFile CONFIGFILE
                        Path to JSON config file.
```

### Description

* `--model`: Selects the GAN architecture to train. Options are:

  * `dcgan`: Deep Convolutional GAN
  * `wgan`: Wasserstein GAN (It is not implemented in this case)

* `--dataset`: Specifies the dataset to use for training. Available options:

  * `chestct`: Chest CT scan dataset
  * `nbia`: NBIA dataset (if available)

* `--configFile`: Optional argument to provide a custom path to a JSON configuration file that contains hyperparameters and training settings. If omitted, the script uses the default `config.json`.

### Example command to train a DCGAN on the chest CT dataset:

```bash
python train.py --model dcgan --dataset chestct
```

## 🚀 Generating New Images

```bash
python generate.py -load_path checkpoints/final_model.pth -num_output 10
````

Parameters:

* `-load_path`: Path to the trained model checkpoint to load.
* `-num_output`: Number of images to generate.
* `-compare`: If enabled, shows a comparison between generated and real images.

```

## 📊 Evaluation and Visualization

### Model Evaluation

You can evaluate a model using the `eval_model.py` script. This script assesses the quality of a GAN model using various metrics such as discriminator and generator accuracy, SSIM, PSNR, and LPIPS.

```bash
python eval_model.py --dataset <dataset> --model_name <model_name>
````

### Arguments:

* `--dataset`: Dataset to use for evaluation. Options include:

  * `nbia` (default)
  * `chestct`
* `--model_name`: Name of the model checkpoint to evaluate. Example: `model_ChestCT.pth`
* `--discarded`: If set, shows discarded metrics info (IS, FID, Precision & Recall for GANs)
* `--configFile`: Path to a custom configuration JSON file

The evaluation results include discriminator accuracy, generator confidence, and image quality metrics such as SSIM, PSNR, and LPIPS.

```

### Training Logs Visualization

You can visualize training logs using the `graphLogs.py` script. This script generates a graph from the training logs, allowing you to observe the model’s performance over time.

#### Usage:

```bash
python graphLogs.py --log_file <log_file.csv>
````

### Arguments:

* `--log_file`: The CSV file containing the training logs. Example: `training_log_dcgan_2025-03-23.csv`.

The generated graph includes discriminator loss, generator loss, and other performance metrics.

---

### Training Logs

Training results can be found in the file `training_log_dcgan_12Feb.csv`, which contains details about the training progress such as loss and accuracy metrics over epochs.

This file can be used as input for `graphLogs.py` to create visualizations of the training outcomes.

```

