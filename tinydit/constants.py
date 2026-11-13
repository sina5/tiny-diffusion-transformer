"""Central constants for this repository.

Scripts and classes read their defaults from here so that one value has one home.
"""

from pathlib import Path

# --- Paths --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"

# --- General ------------------------------------------------------------------
DEFAULT_SEED = 0
BYTES_PER_MIB = 1024 * 1024

# --- CIFAR-10 -----------------------------------------------------------------
CIFAR10_IMAGE_SIZE = 32
CIFAR10_NUM_CLASSES = 10
CIFAR10_CROP_PADDING = 4
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# --- RandAugment --------------------------------------------------------------
RANDAUGMENT_NUM_OPS = 2
RANDAUGMENT_MAGNITUDE = 9

# --- Tiny diffusion model -----------------------------------------------------
# Model: a class-conditional Diffusion Transformer (DiT, Peebles and Xie, 2023)
PATCH_SIZE = 2  # 16x16 = 256 tokens per 32x32 image
DIM = 256
DEPTH = 8
HEADS = 4
MLP_RATIO = 4.0
TIME_FREQ_DIM = 256  # sinusoidal features before the timestep MLP
MAX_PERIOD = 10_000.0
# Diffusion process (Ho et al., 2020)
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
SAMPLE_STEPS = 100  # DDIM steps when sampling
GUIDANCE_SCALE = 2.0  # classifier-free guidance weight
LABEL_DROP = 0.1  # share of training labels replaced by the "no class" token
# Training
EPOCHS = 200
BATCH_SIZE = 128
LR = 3e-4
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 1000
GRAD_CLIP_NORM = 1.0
EMA_DECAY = 0.9999
SNAPSHOT_EVERY = 5  # epochs between sample grids and checkpoints
SAMPLES_PER_CLASS = 8  # one row per class in each sample grid
SAMPLE_SEED = 1234  # the same noise for every snapshot
TRAJECTORY_FRAMES = 10  # denoising steps shown in the trajectory figure
TRAJECTORY_ROWS = 4  # samples (one per class) shown in the trajectory figure
NEIGHBOR_SAMPLES = 10  # samples checked against their nearest training image
GIF_FRAME_MS = 250
GRID_UPSCALE = 3  # nearest-neighbor upscale so 32x32 samples stay sharp on a web page
SMOKE_EPOCHS = 2
SMOKE_SUBSET = 512
SMOKE_SAMPLE_STEPS = 5

# --- Plotting -----------------------------------------------------------------
FIG_SIZE = (16, 9)
FIG_DPI = 150
VIT_COLOR = "#E63946"
