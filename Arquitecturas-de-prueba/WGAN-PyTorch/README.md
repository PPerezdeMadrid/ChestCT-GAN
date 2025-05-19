# WGAN – Wasserstein GAN with Weight Clipping (PyTorch)

This folder contains an implementation of a **Wasserstein GAN (WGAN)** with weight clipping using PyTorch. The WGAN improves upon standard GANs by using a new loss function based on the Earth-Mover (Wasserstein-1) distance, which results in better training stability and gradient behavior.

---

## 📁 Project Structure

```

WGAN/
├── train_wc.py              # Train WGAN with weight clipping
├── generate.py              # Generate images using a trained WGAN model
├── eval_model.py            # Evaluate the trained WGAN using quality metrics
├── config.json              # Training configuration parameters
├── requirements_wgan.txt    # Python dependencies for WGAN
├── wgan.py                  # WGAN model architecture
├── utils.py                 # Utility functions
├── graphLogs.py             # Plot training logs
├── training_log_wgan.csv    # Log file from a training run (generated in the training)              
├── ChestTC.gif              # GIF generated in the training
└── README.md

```

---

## 🔍 What is a WGAN?

A **Wasserstein GAN** replaces the traditional GAN loss with the Wasserstein distance, offering:

- Better gradients for the generator during training
- Increased stability
- No mode collapse (in most cases)

Instead of using a discriminator that outputs probabilities, the **critic** in a WGAN outputs raw scores. Weight clipping is used to enforce the Lipschitz constraint required by the Wasserstein formulation.

---

## 🚀 Getting Started

Install the required Python packages:

```bash
pip install -r requirements_wgan.txt
```

---

## 🏋️‍♂️ Training the Model

Train the WGAN with weight clipping using:

```bash
python train_wc.py
```

The script reads training parameters from `config.json`, including learning rate, batch size, clipping values, and number of critic updates per generator step.

### Sample `config.json`:

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
        "save_epoch": 100,
        "critic_iters": 2
    },
    "model": {
        "path": "model/model_wgan",
        "image_path": "images/images_wgan"
    },
    "datasets":{
        "chestKaggle": "../../../../TFG/ChestCTKaggle/Data/",
        "nbia": "../../src/Pipeline/Data/Data-Transformed"
    }
}
```

---

## 🎨 Generating Images

Use the `generate.py` script to produce new images:

```bash
python generate.py --load_path checkpoints/wgan_final.pth --num_output 10
```

### Arguments:

* `--load_path`: Path to the model checkpoint.
* `--num_output`: Number of images to generate.
* `--compare`: *(Optional)* Display a comparison between real and generated images.

---

## 📊 Evaluating the Model

Evaluate the model’s performance with:

```bash
python eval_model.py --dataset chestct --model_name wgan_final.pth
```

### Available metrics:

* **Discriminator Accuracy**
* **Generator Confidence**
* **SSIM (Structural Similarity Index)**
* **PSNR (Peak Signal-to-Noise Ratio)**
* **LPIPS (Learned Perceptual Image Patch Similarity)**

If you add the `--discarded` flag, it will also display evaluation metrics on discarded or difficult samples.

---


## 📌 Notes

* This implementation uses **weight clipping** (`clip_value`) to enforce the 1-Lipschitz constraint, as proposed in the original WGAN paper.
* Although **WGAN-GP** (with gradient penalty) is known to improve training stability and convergence, **this project focuses exclusively on the clipped WGAN** as part of its academic scope.
* The file `train_gp.py` contains a prototype implementation of WGAN-GP for future experiments, but it is not part of the core submission.


---

## 📚 References

* Arjovsky, M., Chintala, S., & Bottou, L. (2017). *Wasserstein GAN*. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)

