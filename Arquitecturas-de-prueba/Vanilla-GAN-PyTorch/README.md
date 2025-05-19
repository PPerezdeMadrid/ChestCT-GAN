# Vanilla GAN – PyTorch Implementation

This project implements a **Vanilla GAN (Generative Adversarial Network)** using PyTorch. The goal is to generate synthetic images from a given dataset using a basic GAN architecture. This type of GAN consists of a simple fully-connected generator and discriminator, making it ideal for educational purposes or smaller datasets.

## 📁 Project Structure

```

VanillaGAN/
├── train.py             # Train the GAN model using config.json
├── generate.py          # Generate synthetic images using a trained model
├── eval_model.py        # Evaluate the model using standard metrics
├── config.json          # Configuration file for training parameters
├── requirements_vanillaGAN.txt     # Required Python packages
└── README.md        

```

---

## 📌 What is a Vanilla GAN?

A **Vanilla GAN** is the most basic form of a Generative Adversarial Network. It consists of:

- **Generator**: Takes random noise as input and tries to generate realistic images.
- **Discriminator**: Tries to distinguish between real and fake images.

The two networks compete in a minimax game during training, gradually improving each other's performance.

---

## 🚀 Getting Started

Install the required dependencies:

```bash
pip install -r requirements_vanillaGAN.txt
```

---

## 🏋️‍♂️ Training the Model

Train the model using `train.py`:

```bash
python train.py
```

The script will use the configuration specified in `config.json`, which includes parameters like learning rate, batch size, number of epochs, and dataset paths.

### Configuration Example (`config.json`):

```json
{
    "params": {
        "bsize": 128,
        "imsize": 64,
        "nc": 1,
        "nz": 100,
        "ngf": 64,
        "ndf": 64,
        "nepochs": 1000,
        "lr": 0.0002,
        "beta1": 0.9,
        "save_epoch": 10,
        "n_noise":64
    },
    "model": {
        "path": "../../model/model_gan",
        "image_path": "../../images/images_gan"
    }
}
```

---

## 🎨 Generating Images

After training, you can generate synthetic images using:

```bash
python generate.py --load_path checkpoints/final_model.pth --num_output 10
```

### Arguments:

* `--load_path`: Path to the saved model checkpoint.
* `--num_output`: Number of images to generate.

The script will save generated images in the output directory.

---

## 📊 Evaluating the Model

Evaluate the quality of the trained GAN model using:

```bash
python eval_model.py --model_name final_model.pth
```

### Metrics:

* **Discriminator Accuracy**
* **Generator Confidence**
* **SSIM (Structural Similarity Index)**
* **PSNR (Peak Signal-to-Noise Ratio)**
* **LPIPS (Learned Perceptual Image Patch Similarity)**

These metrics help assess both the fidelity and diversity of the generated images.

---

## 📌 Notes

* This is a minimal implementation designed for experimentation and educational purposes.
* You can easily extend it to other datasets or replace the architecture with more advanced GAN variants (e.g., DCGAN, WGAN).

