# tiny-diffusion-transformer

A tiny diffusion model.

This project trains a small diffusion model on CIFAR-10 (32×32 images) and shows how its samples improve as training goes on. The model is a Diffusion Transformer (DiT), which is a ViT that removes noise.

## Method

- **Model** (`tinydit/models/dit.py`): a class-conditional DiT (Peebles and Xie, 2023). It has patch size 2 (256 tokens), width 256, depth 8 and 4 heads, which is 9.8 M parameters. The timestep and the class label control each block through adaptive layer norm. The gates of the adaptive layer norm start at zero ("adaLN-Zero"), so each block starts as the identity.
- **Training** (`tinydit/diffusion.py`): DDPM (Ho et al., 2020). The script adds Gaussian noise at a random timestep (1,000 steps, linear beta schedule from 1e-4 to 0.02), and the model learns to predict that noise. The loss is the mean squared error. For 10% of the batches, the label is replaced by a "no class" token, so one model learns both the conditional and the unconditional prediction.
- **Sampling**: deterministic DDIM (Song et al., 2021) with 100 steps and classifier-free guidance of scale 2. Samples come from an exponential moving average (EMA) of the weights, with decay 0.9999.
- **Optimizer**: AdamW with learning rate 3e-4, no weight decay, a 1,000-step linear warmup, then a constant rate. The batch size is 128. Training runs for 200 epochs with horizontal flips and bfloat16 mixed precision.

## Other methods

The same DiT, data, optimizer, EMA and number of epochs also train with three other methods (`tinydit/methods/`). Only the training target and the sampler change. Continuous times are in [0, 1], where t = 1 is pure noise, and the path from data to noise is a straight line: x_t = (1 − t) x_0 + t · noise.

- **Flow matching** (`flow_matching`, Lipman et al., 2023; Liu et al., 2023): the model predicts the velocity noise − x_0. The training times come from a logit-normal distribution (Esser et al., 2024). Sampling solves the ODE from t = 1 to t = 0 with the Euler or the Heun method.
- **MeanFlow** (`meanflow`, Geng et al., 2025): the model predicts the average velocity over an interval [r, t], so one step can cross the full interval. The target comes from the MeanFlow identity and a Jacobian-vector product. Guidance is part of the training target, so each sampling step is one model call. The DiT gets the times in [0, 1] without the 1000× scale that the other methods use: with that scale, du/dt in the target is very large and training diverges.
- **Shortcut model** (`shortcut`, Frans et al., 2025): the model also gets the step size d. One step of size 2d learns to land where two steps of size d land. This needs no Jacobian-vector product.

The flow-map methods (MeanFlow and shortcut) give the DiT a second time input, the length of the step. It has its own embedding.

## Reward fine-tuning (DDRL)

`scripts/finetune_reward.py` fine-tunes the DDPM model with reinforcement learning (`tinydit/rl/`). It starts from the EMA weights of the DDPM run. The reward is log p(class | image) from a frozen ViT classifier: the "patch 4, 2x epochs" model from the ViT-from-scratch project, with 93.1% test accuracy. Copy its checkpoint to `runs/classifiers/vit_p4_long.pt`.

- **Rollouts**: stochastic DDIM (eta = 1) with 20 steps and no guidance. Each iteration has 16 labels with 8 rollouts each. The reward is normalized inside each group of 8 (the advantage).
- **Loss**: −advantage × log p(x_s | x_t), on every second step (Ye et al., 2025, Algorithm 1).
- **Regularizer**: `none` (reward only), `reverse_kl` (KL from each step to the frozen base model), or `data` (DDRL: the normal diffusion loss on real training images, with weight beta).
- **Checks for reward hacking**: a ResNet-18 judge that the fine-tuning never sees (`scripts/train_judge.py`), FID, and the diffusion loss on test images.

```bash
uv run python scripts/train_judge.py
uv run python scripts/finetune_reward.py --base-only
uv run python scripts/finetune_reward.py --regularizer data --beta 0.1   # writes runs/ddrl/data_beta0.1/
```

## FID

`scripts/evaluate_fid.py` computes FID from 10,000 class-balanced samples against the 50,000 training images (FID-10k). It uses the Inception network from pytorch-fid. The cost of each setting is the number of function evaluations (NFE), which is the number of DiT calls for each image. A guided step counts as two.

## Setup

You need [uv](https://docs.astral.sh/uv/) and an NVIDIA driver that supports CUDA 12.8.

```bash
uv sync --group dev
uv run pytest
```

`uv sync` installs PyTorch with CUDA 12.8 from the PyTorch package index. The scripts download the dataset to `data/` on first use. They write results, charts and checkpoints to `runs/`. Git ignores `data/` and `runs/`.

## Run

```bash
# Check the pipeline (2 epochs on 512 images, 5 sampling steps)
uv run python scripts/train_diffusion.py --smoke

# Full run (resumes from the last checkpoint if stopped)
uv run python scripts/train_diffusion.py

# Redraw the figures from the saved run
uv run python scripts/train_diffusion.py --figures-only

# Another method (writes to runs/<method>/)
uv run python scripts/train_diffusion.py --method flow_matching   # or meanflow, shortcut

# FID-10k at several step counts (appends to fid.json in the run folder)
uv run python scripts/evaluate_fid.py --method flow_matching --steps 1 2 4 8 16 32 64 128
uv run python scripts/evaluate_fid.py --method flow_matching --solver heun --steps 4 8 16

# Charts and equal-NFE sample grids for the blog posts (writes runs/blog_figures/)
uv run python scripts/make_blog_figures.py
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
- **FID-10k** (`runs/fid.json`): 83.4 at 8 NFE, 30.1 at 32 NFE and 19.9 at 256 NFE (DDIM with 4, 16 and 128 steps). With 1 or 2 steps, DDIM gives noise (FID above 350).

### Other methods

Each method has one run of 200 epochs. FID-10k is from `runs/<method>/fid.json`.

| Method | Time per epoch | FID-10k at 2 NFE | at 8 NFE | at 32 NFE | Best (NFE) |
| --- | --- | --- | --- | --- | --- |
| DDPM (DDIM) | 25.9 s | 360.9 | 83.4 | 30.1 | 19.9 (256) |
| Flow matching (Euler) | 25.9 s | 310.0 | 38.9 | 20.2 | 14.0 (256) |
| MeanFlow | 50.5 s | 17.5 | 16.1 | 17.7 | 15.0 (16) |
| Shortcut model | 31.0 s | 40.5 | 18.2 | 16.8 | 16.7 (64) |

MeanFlow gives 39.9 with one model call. The first MeanFlow run scaled the time input by 1000; its loss diverged (`runs/meanflow_time1000_diverged/`).

### Reward fine-tuning

The final FID-10k of the base model (stochastic DDIM, 20 steps, no guidance) is 51.9. After 300 iterations at learning rate 1e-5, the data regularizer with beta 0.01 gives 38.6 (seed 0) and 38.2 (seed 1), and reward only gives 47.3 and 43.7. At learning rate 1e-4, reward only collapses (FID 379.4), and the data regularizer with beta 0.01 hacks the reward: the judge accuracy drops from 50.8% to 45.7% while the reward-ViT accuracy rises. The values are in `runs/ddrl/*/final.json`.

## Layout

```
tinydit/               package
  constants.py         every default value (epochs, learning rates, paths)
  models/dit.py        the Diffusion Transformer
  diffusion.py         noise schedule, DDIM sampler, EMA
  methods/             DDPM, flow matching, MeanFlow, shortcut model (loss + sampler)
  evaluation/fid.py    FID
  rl/                  reward fine-tuning: stochastic sampler, rewards, DDRL, judge
  models/classifiers.py  reward ViT and judge ResNet-18
  data.py              CIFAR-10 pipeline
  plotting.py          chart helpers
  experiments/         training loop, snapshots, figures, FID sweep, blog figures
scripts/               entry scripts
tests/                 pytest suite (CPU only, no downloads)
```

## License

The code is licensed under the [Apache License 2.0](LICENSE).
