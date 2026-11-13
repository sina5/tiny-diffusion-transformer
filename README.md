# tiny-diffusion-transformer

A tiny diffusion model.

This project trains a small diffusion model on CIFAR-10 (32×32 images) and shows how its samples improve as training goes on. The model is a Diffusion Transformer (DiT), which is a ViT that removes noise.

## Method

- **Model** (`tinydit/models/dit.py`): a class-conditional DiT (Peebles and Xie, 2023). It has patch size 2 (256 tokens), width 256, depth 8 and 4 heads, which is 9.8 M parameters. The timestep and the class label control each block through adaptive layer norm. The gates of the adaptive layer norm start at zero ("adaLN-Zero"), so each block starts as the identity.
- **Training** (`tinydit/diffusion.py`): DDPM (Ho et al., 2020). The script adds Gaussian noise at a random timestep (1,000 steps, linear beta schedule from 1e-4 to 0.02), and the model learns to predict that noise. The loss is the mean squared error. For 10% of the batches, the label is replaced by a "no class" token, so one model learns both the conditional and the unconditional prediction.
- **Sampling**: deterministic DDIM (Song et al., 2021) with 100 steps and classifier-free guidance of scale 2. Samples come from an exponential moving average (EMA) of the weights, with decay 0.9999.
- **Optimizer**: AdamW with learning rate 3e-4, no weight decay, a 1,000-step linear warmup, then a constant rate. The batch size is 128. Training runs for 200 epochs with horizontal flips and bfloat16 mixed precision.

## Setup

You need [uv](https://docs.astral.sh/uv/) and an NVIDIA driver that supports CUDA 12.8.

```bash
uv sync --group dev
uv run pytest
```

`uv sync` installs PyTorch with CUDA 12.8 from the PyTorch package index. The scripts download the dataset to `data/` on first use. They write results, charts and checkpoints to `runs/`. The repository includes the results and charts of the full runs in `runs/`. Git ignores `data/`, the model weights and the `smoke/` folders.

## Run

```bash
# Check the pipeline (2 epochs on 512 images, 5 sampling steps)
uv run python scripts/train_diffusion.py --smoke

# Full run (resumes from the last checkpoint if stopped)
uv run python scripts/train_diffusion.py

# Redraw the figures from the saved run
uv run python scripts/train_diffusion.py --figures-only
```

## Output

In `runs/`:

- `snapshots/epoch_NNNN.pt`: every 5 epochs, 80 samples (8 per class) from the same fixed noise.
- `samples_final.png`: the last snapshot. Each row is a class, in CIFAR-10 order (airplane to truck).
- `samples_over_training.gif`: all snapshots as an animation.
- `progress.png`: the same noise at 6 points during training.
- `denoising_path.png`: the predicted clean image at 10 points during sampling.
- `nearest_training_images.png`: each sample next to the training image that is closest to it in pixel space. This checks that the model does not copy training images.
- `loss.png`, `history.json`: the loss for each epoch, the time, the peak VRAM and the parameter count.
- `checkpoint.pt`: model, EMA, optimizer and scheduler state. Model weights are not stored in Git.

## Results

One run of 200 epochs: 86.5 minutes of training on an RTX 3080 Ti (26 s per epoch), with a peak VRAM of 3,595 MiB. The model has 9.8 M parameters.

- **Samples**: airplanes, cars, horses, ships and trucks have clear shapes and plausible backgrounds. Cats, dogs, birds and frogs often have the right colors and texture but a blurred or broken shape.
- **Progress**: the EMA samples are still noise at epoch 45. Objects appear by epoch 85, and the samples change little after epoch 120. The training loss drops much earlier (0.033 at epoch 25, 0.030 at epoch 200). Part of the delay comes from the EMA: with decay 0.9999, the averaged weights trail the current weights by about 10,000 steps, which is about 26 epochs.
- **Memorization check**: the nearest training image in pixel space is a different image for each of the 10 samples checked. This is a small check, not a proof.
- **Quality metric**: the project does not compute FID, so the results are visual only.

## Layout

```
tinydit/               package
  constants.py         every default value (epochs, learning rates, paths)
  models/dit.py        the Diffusion Transformer
  diffusion.py         noise schedule, DDIM sampler, EMA
  data.py              CIFAR-10 pipeline
  plotting.py          chart helpers
  experiments/         training loop, snapshots, figures
scripts/               entry script
tests/                 pytest suite (CPU only, no downloads)
```

## License

The code is licensed under the [Apache License 2.0](LICENSE).
