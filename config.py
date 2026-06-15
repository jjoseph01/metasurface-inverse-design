# config.py
# Central place for every knob in the project. If you want to change how things
# train, this is almost always the only file you need to touch.
import os
import torch
import matplotlib

matplotlib.use('Agg')  # we only ever save figures to disk, never show them

# --- 0. Core configuration ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 'CONSTANT_TARGET' trains a generator for one fixed absorbance target at a time.
# 'CONDITIONAL'     trains a single generator conditioned on any target you hand it.
TRAINING_MODE = 'CONSTANT_TARGET'

# --- 1. Data paths ---
# Everything is relative to the repo root so this runs on any machine. Drop your
# data in a Data/ folder next to this file (see the README for the exact layout).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER_PATH = os.path.join(PROJECT_ROOT, "Data", "Data_Generated_Images")
METADATA_FILE = os.path.join(PROJECT_ROOT, "Data", "metasurface_absorbance_compiled_final.csv")
TARGET_RESPONSES_FILE = os.path.join(PROJECT_ROOT, "Data", "target_responses.csv")

# If a checkpoint exists here, simulator pre-training is skipped and we load it.
# Set to None to always train the simulator from scratch.
PRETRAINED_SIMULATOR_PATH = os.path.join(PROJECT_ROOT, "best_simulator_model.pth")

# --- 2. Image and data dimensions ---
CURRENT_GAN_IMAGE_SIZE = 64
SIMULATOR_IMAGE_SIZE = 64
CHANNELS = 1
NUM_ANGLES = 15          # absorbance is measured at 15 incidence angles (0 to 14*pi/30)
TEST_SPLIT_SIZE = 0.10

# --- 3. Model parameters ---
LATENT_DIM = 128
GF = 64                  # generator feature width
DF = 64                  # critic feature width
SIM_NDF = 256            # simulator feature width
DROPOUT_RATE_SIM = 0.3   # 0.3 worked best for us

# --- 4. Training length ---
BATCH_SIZE = 128
WORKERS = 2
NUM_EPOCHS_SIMULATOR = 500 if TRAINING_MODE == 'CONSTANT_TARGET' else 300
NUM_EPOCHS_GAN = 1 if TRAINING_MODE == 'CONSTANT_TARGET' else 15000

# --- Simulator loss & optimizer ---
LAMBDA_GRAD = 1          # weight on the shape (gradient) term of the simulator loss
WEIGHT_DECAY_SIMULATOR = 1e-4

# --- 5. GAN optimizer and loss ---
LEARNING_RATE_SIM = 0.0002
INITIAL_LR_G = 0.0001
INITIAL_LR_C = 0.00005
BETA1_ADAM = 0.0
BETA2_ADAM = 0.9
N_CRITIC = 5             # critic updates per generator update (WGAN-GP)
GP_WEIGHT = 10.0         # gradient-penalty weight
LAMBDA_SIM_LOSS = 0.000025  # how hard the simulator pushes the generator toward the target

# --- 6. Output directory ---
OUTPUT_DIR_BASE = "output_constant_target" if TRAINING_MODE == 'CONSTANT_TARGET' else "output_conditional"

# --- 7. Evaluation ---
NUM_IMAGES_PER_ABSORPTION_TEST = 5
NUM_EVAL_IMAGES_FOR_CONSTANT_TARGET = 100
