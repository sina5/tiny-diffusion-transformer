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

# --- Generative methods -------------------------------------------------------
# Continuous methods use t in [0, 1] (t = 1 is pure noise). The timestep embedding was built
# for DDPM's 0..999 range, so continuous times are scaled up by this factor before embedding.
TIME_SCALE = 1000.0
# Flow matching (Lipman et al., 2023; Liu et al., 2023), logit-normal t (Esser et al., 2024)
FLOW_LOGIT_MEAN = 0.0
FLOW_LOGIT_STD = 1.0
FLOW_SAMPLE_STEPS = 100
# MeanFlow (Geng et al., 2025)
MEANFLOW_LOGIT_MEAN = -0.4
MEANFLOW_LOGIT_STD = 1.0
MEANFLOW_UNEQUAL_SHARE = 0.25  # share of the batch with r < t; the rest is plain flow matching
MEANFLOW_ADAPTIVE_POWER = 1.0  # loss weight 1 / (error + c)^p
MEANFLOW_ADAPTIVE_EPS = 1e-3
MEANFLOW_SAMPLE_STEPS = 1
MEANFLOW_TIME_SCALE = 1.0  # times go to the DiT unscaled; 1000 makes the du/dt target diverge
# Shortcut models (Frans et al., 2025)
SHORTCUT_BOOTSTRAP_SHARE = 0.25  # share of the batch trained on self-consistency targets
SHORTCUT_BASE_STEPS = 128  # finest step size is 1 / 128
SHORTCUT_SAMPLE_STEPS = 4

# --- Reward fine-tuning (DDRL, Ye et al., 2025) -------------------------------
CLASSIFIER_DIR = RUNS_DIR / "classifiers"
REWARD_CHECKPOINT = "vit_p4_long.pt"  # from the ViT-from-scratch post (93.1% test accuracy)
JUDGE_CHECKPOINT = "resnet18_judge.pt"
JUDGE_EPOCHS = 30
JUDGE_BATCH_SIZE = 256
JUDGE_LR = 0.1
JUDGE_MOMENTUM = 0.9
JUDGE_WEIGHT_DECAY = 5e-4
JUDGE_WORKERS = 4
RL_SAMPLE_STEPS = 20  # stochastic DDIM (eta = 1) steps per rollout
RL_TRAIN_EVERY = 2  # train on every second transition, as DDRL does
RL_GROUP_SIZE = 8  # rollouts per label; the advantage is normalized inside each group
RL_GROUPS = 16  # labels per iteration, so 128 rollouts
RL_ITERATIONS = 300
RL_LR = 1e-5
RL_DATA_BETA = 0.1  # weight of the diffusion loss on real images (DDRL)
RL_KL_BETA = 0.1  # weight of the reverse KL to the frozen base model (the KL baseline)
RL_EVAL_EVERY = 25
RL_EVAL_SAMPLES = 2000
RL_EVAL_SEED = 999
RL_FINAL_FID_SAMPLES = 10_000
ADVANTAGE_EPS = 1e-6

# --- FID ----------------------------------------------------------------------
FID_SAMPLES = 10_000
FID_BATCH = 500
FID_SEED = 4321
FID_DIMS = 2048  # Inception pool3 features
FID_REFERENCE_FILE = "fid_reference_cifar10_train.npz"

# --- Plotting -----------------------------------------------------------------
FIG_SIZE = (16, 9)
FIG_DPI = 150
VIT_COLOR = "#E63946"
# Categorical slots in fixed order (blue, orange, aqua, yellow, magenta), checked for color-blind separation
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
REFERENCE_COLOR = "#52514e"  # the base model and other reference lines

# --- Blog figures -------------------------------------------------------------
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
BLOG_FIGURES_DIR = RUNS_DIR / "blog_figures"
COMPARE_SEED = 2024  # the same noise for every panel of a comparison grid
COMPARE_PER_CLASS = 4  # samples per class in each panel
