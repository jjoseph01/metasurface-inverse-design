# main.py
import torch
import os
import time
import pandas as pd
from datetime import datetime

from config import (
    DEVICE, IMAGE_FOLDER_PATH, METADATA_FILE, TARGET_RESPONSES_FILE,
    CURRENT_GAN_IMAGE_SIZE, SIMULATOR_IMAGE_SIZE, NUM_ANGLES, LATENT_DIM,
    NUM_EPOCHS_SIMULATOR, NUM_EPOCHS_GAN, BATCH_SIZE, WORKERS,
    GP_WEIGHT, N_CRITIC, LAMBDA_SIM_LOSS,
    OUTPUT_DIR_BASE, TRAINING_MODE,
    NUM_IMAGES_PER_ABSORPTION_TEST,
    NUM_EVAL_IMAGES_FOR_CONSTANT_TARGET, PRETRAINED_SIMULATOR_PATH
)

from data_loader import get_dataloaders, resize_for_simulator
from models import Generator, ConditionalCritic, UnconditionalCritic, RobustSimulatorCNN
from losses_optimizers import get_optimizers_and_schedulers, get_simulator_criterion
from training_loops import pretrain_simulator, train_gan_conditional, train_gan_constant_target
# --- UPDATED IMPORTS ---
from graph_output import (
    plot_simulator_loss, plot_gan_losses_and_lr,
    plot_simulator_final_evaluation,
    plot_gan_constant_target_performance, # Renamed from plot_gan_final_performance
    plot_gan_evaluation_with_dtw # New DTW evaluation function
)
from utils import weights_init


def check_gpu():
    """Checks and prints the GPU status for confirmation."""
    print("--- Checking GPU Status ---")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA is available. Found {gpu_count} GPU(s).")
        print(f"Using GPU: {gpu_name}")
        print(f"PyTorch is using device: {DEVICE}")
    else:
        print(">>> WARNING: CUDA is NOT available. PyTorch will run on the CPU. <<<")
    print("-" * 27)


class RunManager:
    """
    Manages file paths for a single execution run, ensuring a clean,
    timestamped directory structure for all outputs.
    """
    def __init__(self, base_output_dir, training_mode):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"run_{training_mode}_{self.timestamp}"
        self.run_dir = os.path.join(base_output_dir, self.run_name)
        
        self.simulator_output_dir = os.path.join(self.run_dir, "simulator_pretraining")
        self.targets_base_dir = os.path.join(self.run_dir, "target_specific_training")

        self._create_initial_dirs()
        self.summary_file_path = os.path.join(self.run_dir, "run_summary.txt")
        print(f"--- Run initialized. All outputs will be saved in: {self.run_dir} ---")

    def _create_initial_dirs(self):
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.simulator_output_dir, exist_ok=True)
        os.makedirs(self.targets_base_dir, exist_ok=True)

    def log_summary(self, message):
        with open(self.summary_file_path, 'a') as f:
            f.write(message + "\n")

    def get_target_dirs(self, target_label):
        target_dir = os.path.join(self.targets_base_dir, target_label)
        gan_training_dir = os.path.join(target_dir, "gan_training_process")
        checkpoints_dir = os.path.join(gan_training_dir, "checkpoints")
        training_images_dir = os.path.join(gan_training_dir, "images")
        final_eval_dir = os.path.join(target_dir, "final_evaluation")
        
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(gan_training_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(training_images_dir, exist_ok=True)
        os.makedirs(final_eval_dir, exist_ok=True)
        
        return {
            "target_base": target_dir,
            "gan_training": gan_training_dir,
            "checkpoints": checkpoints_dir,
            "training_images": training_images_dir,
            "final_evaluation": final_eval_dir
        }

def main():
    check_gpu()

    # --- 0. Setup Run Manager and Initial Logging ---
    run_manager = RunManager(OUTPUT_DIR_BASE, TRAINING_MODE)
    run_manager.log_summary(f"Run started at: {run_manager.timestamp}")
    run_manager.log_summary(f"Device: {DEVICE}")
    run_manager.log_summary(f"Training Mode: {TRAINING_MODE}")
    run_manager.log_summary(f"GAN Image Size: {CURRENT_GAN_IMAGE_SIZE}, Simulator Image Size: {SIMULATOR_IMAGE_SIZE}")
    run_manager.log_summary(f"Epochs (Simulator): {NUM_EPOCHS_SIMULATOR}, Epochs (GAN): {NUM_EPOCHS_GAN}")
    run_manager.log_summary(f"Batch Size: {BATCH_SIZE}, Latent Dim: {LATENT_DIM}")

    # --- 1. Data Loading ---
    print("\n--- 1. Setting up Datasets and DataLoaders ---")
    sim_train_loader, sim_test_loader, gan_loader, full_dataset = get_dataloaders(
        mode=TRAINING_MODE,
        image_folder_path=IMAGE_FOLDER_PATH,
        metadata_file=METADATA_FILE
    )
    if not gan_loader or not sim_train_loader:
        print("Error: Data could not be loaded. Exiting."); return

    num_batches_per_epoch = len(gan_loader)
    decay_steps_total = num_batches_per_epoch * NUM_EPOCHS_GAN
    print(f"Data loading complete. Total dataset samples: {len(full_dataset)}")

    # --- 2. Instantiating Simulator Model (Load or Train) ---
    print("\n--- 2. Instantiating Simulator Model (Load or Train) ---")
    simulator_model = RobustSimulatorCNN().to(DEVICE)
    simulator_model.apply(weights_init)
    
    criterion_S = get_simulator_criterion()
    
    if PRETRAINED_SIMULATOR_PATH and os.path.exists(PRETRAINED_SIMULATOR_PATH):
        print(f"✅ Found pre-trained model. Loading from: {PRETRAINED_SIMULATOR_PATH}")
        checkpoint = torch.load(PRETRAINED_SIMULATOR_PATH, map_location=DEVICE)
        simulator_model.load_state_dict(checkpoint['model_state_dict'])
        print("Simulator model loaded successfully.")
        train_losses_sim, test_losses_sim = [], []
    else:
        print("No valid pre-trained simulator path found. Starting a new training session...")
        run_manager.log_summary("Training new simulator model from scratch.")
        
        _, _, optimizer_sim, _, _, scheduler_sim = get_optimizers_and_schedulers(
            Generator(), UnconditionalCritic(), simulator_model, decay_steps_total
        )
        train_losses_sim, test_losses_sim = pretrain_simulator(
            simulator_model, sim_train_loader, sim_test_loader, optimizer_sim, scheduler_sim,
            criterion_S, NUM_EPOCHS_SIMULATOR, DEVICE, run_manager.simulator_output_dir
        )
    
    # --- 3. Evaluate and Plot Simulator Performance ---
    print("\n--- 3. Plotting Simulator Performance ---")
    
    if train_losses_sim:
        plot_simulator_loss(train_losses_sim, test_losses_sim, run_manager.simulator_output_dir, sim_test_loader is not None)
    
    plot_simulator_final_evaluation(
        simulator_model, sim_test_loader, DEVICE, run_manager.simulator_output_dir, NUM_ANGLES
    )
    
    simulator_model.eval() 
    [p.requires_grad_(False) for p in simulator_model.parameters()]
    print("Simulator model is ready and frozen for GAN training.")

    # --- 4. Load Target Responses ---
    print("\n--- 4. Loading Target Responses for GAN Training ---")
    try:
        target_responses_df = pd.read_csv(TARGET_RESPONSES_FILE)
        all_target_responses = [torch.tensor(row.values.astype(float), dtype=torch.float32).to(DEVICE)
                                  for _, row in target_responses_df.iterrows()]
        print(f"Loaded {len(all_target_responses)} target responses for sequential training.")
    except Exception as e:
        print(f"Error loading target responses CSV: {e}"); return
    
    # --- 5. Loop through each target response for GAN training ---
    for target_idx, desired_response_single in enumerate(all_target_responses):
        # The response from file might not have a batch dimension, so add it.
        if desired_response_single.dim() == 1:
            desired_response = desired_response_single.unsqueeze(0)
        else:
            desired_response = desired_response_single

        current_target_label = f"Target_{target_idx+1:02d}"
        print(f"\n{'='*60}\n--- Starting GAN Training for {current_target_label} ---\n{'='*60}")
        run_manager.log_summary(f"\nProcessing {current_target_label}...")
        
        target_dirs = run_manager.get_target_dirs(current_target_label)

        print(f"\n--- 5.1 Instantiating GAN Models for {current_target_label} ---")
        netG = Generator().to(DEVICE)
        netC = UnconditionalCritic().to(DEVICE) if TRAINING_MODE == 'CONSTANT_TARGET' else ConditionalCritic().to(DEVICE)
        netG.apply(weights_init); netC.apply(weights_init)

        print(f"\n--- 5.2 Setting up Optimizers and Schedulers for {current_target_label} ---")
        optimizerG, optimizerC, _, scheduler_G, scheduler_C, _ = get_optimizers_and_schedulers(
            netG, netC, None, decay_steps_total
        )

        print(f"\n--- 5.3 Starting GAN Training Loop for {current_target_label} ---")
        fixed_noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1, device=DEVICE)

        if TRAINING_MODE == 'CONSTANT_TARGET':
            G_losses, C_losses, Sim_losses_G_side, logged_lr_g, logged_lr_c = train_gan_constant_target(
                netG, netC, simulator_model, gan_loader, optimizerG, optimizerC, scheduler_G, scheduler_C,
                criterion_S, NUM_EPOCHS_GAN, N_CRITIC, GP_WEIGHT, LAMBDA_SIM_LOSS, DEVICE, 
                target_dirs, fixed_noise, desired_response, resize_for_simulator
            )
        else: # Assumes 'CONDITIONAL' mode
            sample_batch = next(iter(gan_loader))
            fixed_absorbance_for_gen = sample_batch[1][:fixed_noise.size(0)].to(DEVICE)
            G_losses, C_losses, Sim_losses_G_side, logged_lr_g, logged_lr_c = train_gan_conditional(
                netG, netC, simulator_model, gan_loader, optimizerG, optimizerC, scheduler_G, scheduler_C,
                criterion_S, NUM_EPOCHS_GAN, N_CRITIC, GP_WEIGHT, LAMBDA_SIM_LOSS, DEVICE, 
                target_dirs, fixed_noise, fixed_absorbance_for_gen, num_batches_per_epoch, resize_for_simulator
            )
        
        print(f"--- GAN Training for {current_target_label} Finished. ---")

        print(f"\n--- 5.4 Final Evaluation for {current_target_label} ---")
        
        plot_gan_losses_and_lr(G_losses, C_losses, Sim_losses_G_side, logged_lr_g, logged_lr_c, target_dirs["final_evaluation"])

        # --- UPDATED EVALUATION LOGIC ---
        # The new DTW evaluation function is more robust for evaluating performance against specific targets.
        # It replaces the previous separate calls for constant and conditional modes.
        plot_gan_evaluation_with_dtw(
            netG=netG,
            simulator_model=simulator_model,
            target_response=desired_response.squeeze(0), # Ensure it's a 1D tensor for DTW
            device=DEVICE,
            output_dir=target_dirs["final_evaluation"],
            run_name=current_target_label,
            resize_for_simulator=resize_for_simulator,
            num_angles=NUM_ANGLES,
            latent_dim=LATENT_DIM,
            num_geometries=NUM_EVAL_IMAGES_FOR_CONSTANT_TARGET # Use this config for number of test images
        )

        print(f"\n--- Pipeline for {current_target_label} Finished. Outputs saved in: {target_dirs['target_base']} ---")

    print(f"\n--- All Target Responses Processed. Full Pipeline Finished. ---")
    run_manager.log_summary(f"Run finished successfully at: {datetime.now().strftime('%Y%m%d_%H%M%S')}")

if __name__ == "__main__":
    main()
