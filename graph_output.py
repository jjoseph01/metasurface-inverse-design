# graphing.py
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from PIL import Image
import torchvision.utils as vutils
from data_loader import DiverseMetasurfaceDataset
from dtw import dtw # Requires: pip install dtw-python

# --- Universal Plotting & Utility Functions ---

def plot_simulator_loss(train_losses, test_losses, output_dir, has_test_data):
    """Plots and saves the simulator's training and testing loss curves."""
    plt.figure(figsize=(10, 5))
    plt.title("Simulator Model Loss During Pre-training")
    plt.plot(train_losses, label="Training Loss")
    if has_test_data and test_losses:
        plt.plot(test_losses, label="Testing Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "simulator_loss_curve.png"))
    plt.close()
    print(f"Simulator loss curve saved to {output_dir}")

def plot_gan_losses_and_lr(g_losses, c_losses, sim_losses, lr_g, lr_c, output_dir):
    """Plots and saves the WGAN-GP's losses and learning rate schedules."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("GAN Losses During Training")
    plt.plot(g_losses, label="G Loss (Combined)")
    plt.plot(c_losses, label="C Loss")
    if sim_losses:
        plt.plot(sim_losses, label="Sim Loss (on G images)", alpha=0.7, linestyle='--')
    plt.xlabel("Iterations"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(lr_g, label="LR (Generator)")
    plt.plot(lr_c, label="LR (Critic)", linestyle='--')
    plt.title("Learning Rate Schedule")
    plt.xlabel("Iterations"); plt.ylabel("Learning Rate"); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "gan_loss_and_lr_curve.png"))
    plt.close()
    print(f"GAN loss and LR curve saved to {output_dir}")

def calculate_absorbance_accuracy(real_abs, pred_abs):
    """Calculates accuracy based on the Root Mean Squared Error (RMSE)."""
    if not isinstance(real_abs, torch.Tensor): real_abs = torch.tensor(real_abs)
    if not isinstance(pred_abs, torch.Tensor): pred_abs = torch.tensor(pred_abs)
    mse = torch.mean((real_abs - pred_abs) ** 2)
    rmse = torch.sqrt(mse)
    # Clamp accuracy at 0
    accuracy = max(0, 100.0 * (1.0 - rmse.item()))
    return accuracy, rmse.item()

def get_angle_labels(num_angles, is_radians=False):
    """Generates labels for the x-axis of the absorbance plots."""
    if is_radians:
        return [f'{k}π/30' for k in range(num_angles)]
    return [f'{k}' for k in range(num_angles)]

# --- Simulator Evaluation Functions ---

def plot_simulator_final_evaluation(simulator_model, test_dataloader, device, output_dir, num_angles):
    """Performs a final, detailed evaluation of the trained simulator on the test set."""
    if not test_dataloader:
        print("Skipping simulator final evaluation: No test data provided.")
        return

    print("\n--- Starting Final Simulator Evaluation on Test Set ---")
    simulator_model.eval()
    
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    class_metrics = defaultdict(lambda: {'mses': [], 'maes': []})
    all_test_targets, all_test_predictions, all_test_filenames = [], [], []

    # Access the original dataset and indices from the Subset
    original_test_dataset = test_dataloader.dataset.dataset
    test_indices_list = test_dataloader.dataset.indices

    with torch.no_grad():
        for i in range(len(test_indices_list)):
            original_idx = test_indices_list[i]
            image, target = original_test_dataset[original_idx]
            image, target = image.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
            output = simulator_model(image)
            
            original_filename = original_test_dataset.metadata_df.iloc[original_idx][original_test_dataset.filename_col]
            # Use the image_path_map for robust path finding
            full_image_path = original_test_dataset.image_path_map.get(os.path.basename(original_filename))
            
            if not full_image_path: continue

            class_name = os.path.split(os.path.dirname(full_image_path))[-1]
            class_metrics[class_name]['mses'].append(criterion_mse(output, target).item())
            class_metrics[class_name]['maes'].append(criterion_mae(output, target).item())
            all_test_targets.append(target.cpu().numpy())
            all_test_predictions.append(output.cpu().numpy())
            all_test_filenames.append(full_image_path)

    all_targets_np = np.concatenate(all_test_targets, axis=0)
    all_predictions_np = np.concatenate(all_test_predictions, axis=0)

    print("\n--- Final Test Set Metrics (Per Geometry Class) ---")
    for class_name, metrics in sorted(class_metrics.items()):
        avg_mse = np.mean(metrics['mses']); avg_rmse = np.sqrt(avg_mse); avg_mae = np.mean(metrics['maes'])
        print(f"Class: {class_name} ({len(metrics['mses'])} samples) -> Avg RMSE: {avg_rmse:.4f}, Avg MAE: {avg_mae:.4f}")

    overall_mse = np.mean((all_targets_np - all_predictions_np)**2); overall_rmse = np.sqrt(overall_mse)
    overall_mae = np.mean(np.abs(all_targets_np - all_predictions_np))
    print("\n--- Overall Test Set Metrics ---")
    print(f"Average MSE: {overall_mse:.6f}, Average RMSE: {overall_rmse:.6f}, Average MAE: {overall_mae:.6f}")

    plots_save_dir = os.path.join(output_dir, "individual_test_plots")
    os.makedirs(plots_save_dir, exist_ok=True)
    print(f"\nGenerating individual test sample plots in '{plots_save_dir}'...")
    
    angle_indices = np.arange(num_angles)
    x_labels = get_angle_labels(num_angles)

    for i in range(len(all_targets_np)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        original_image = Image.open(all_test_filenames[i]).resize((128, 128))
        ax1.imshow(original_image, cmap='gray')
        ax1.set_title(f"Metasurface:\n{os.path.basename(all_test_filenames[i])}")
        ax1.axis('off')
        ax2.plot(angle_indices, all_targets_np[i], 'bo-', label='Target Absorbance')
        ax2.plot(angle_indices, all_predictions_np[i], 'ro--', label='Predicted Absorbance')
        ax2.set_xticks(angle_indices); ax2.set_xticklabels(x_labels)
        ax2.set_xlabel("Angle Index (0-14)"); ax2.set_ylabel("Absorbance")
        ax2.set_title(f"Absorbance vs. Angle (MAE: {np.mean(np.abs(all_targets_np[i] - all_predictions_np[i])):.4f})")
        ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6); ax2.set_ylim(-0.1, 1.1)
        plt.tight_layout()
        plot_filename = f"test_{os.path.basename(all_test_filenames[i]).replace('.', '_')}.png"
        plt.savefig(os.path.join(plots_save_dir, plot_filename))
        plt.close(fig)

    print(f"Finished saving {len(all_targets_np)} individual test plots.")
    simulator_model.train()

# --- GAN Evaluation Functions ---

def plot_gan_constant_target_performance(netG, simulator_model, desired_response, device, output_dir,
                                         num_test_images, resize_func, num_angles, latent_dim):
    """Evaluates GAN performance for a single, constant target response."""
    print("\n--- Starting GAN Evaluation for Constant Target ---")
    netG.eval(); simulator_model.eval()
    images_save_dir = os.path.join(output_dir, "generated_images_constant_target")
    os.makedirs(images_save_dir, exist_ok=True)
    
    fixed_noise = torch.randn(num_test_images, latent_dim, 1, 1, device=device)
    desired_response_batch = desired_response.repeat(num_test_images, 1)
    with torch.no_grad():
        generated_images = netG(fixed_noise, desired_response_batch).detach()
        predicted_absorbance = simulator_model(resize_func(generated_images)).cpu()
    
    vutils.save_image(generated_images, os.path.join(images_save_dir, 'all_generated_samples.png'), normalize=True, nrow=8)
    print(f"Grid of final generated images saved to {images_save_dir}")
    
    num_plots = min(num_test_images, 8)
    fig, axes = plt.subplots(num_plots, 2, figsize=(12, 4 * num_plots), squeeze=False)
    fig.suptitle("GAN Final Performance: Generated Images and Their Absorbance", fontsize=16)
    x_ticks = np.arange(num_angles)
    x_labels = get_angle_labels(num_angles)
    accuracies = []
    for i in range(num_plots):
        img_display = (generated_images[i] * 0.5 + 0.5).clamp(0, 1).cpu().squeeze().numpy()
        axes[i, 0].imshow(img_display, cmap='gray'); axes[i, 0].set_title(f"Generated Image {i+1}"); axes[i, 0].axis('off')
        accuracy, _ = calculate_absorbance_accuracy(desired_response.cpu(), predicted_absorbance[i])
        accuracies.append(accuracy)
        axes[i, 1].plot(x_ticks, predicted_absorbance[i].numpy(), 'x--', label='Predicted Absorbance', color='red')
        axes[i, 1].plot(x_ticks, desired_response.cpu().squeeze().numpy(), 'o-', label='Desired Absorbance', color='blue', alpha=0.7)
        axes[i, 1].set_ylim(0, 1.1); axes[i, 1].set_title(f"Absorbance Comparison (Acc: {accuracy:.2f}%)"); axes[i, 1].legend()
        axes[i, 1].set_xlabel("Angle Index"); axes[i, 1].set_ylabel("Absorbance"); axes[i, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, 'gan_constant_target_summary.png')
    plt.savefig(plot_path); plt.close(fig)
    print(f"GAN constant target performance summary saved to {plot_path}")
    print(f"Average accuracy over {num_plots} samples: {np.mean(accuracies):.2f}%")
    netG.train(); simulator_model.train()


def plot_gan_evaluation_with_dtw(netG, simulator_model, target_response, device, output_dir,
                                 run_name, resize_for_simulator, num_angles, latent_dim, num_geometries=5):
    """
    Generates geometries for a target, ranks them using DTW, and plots summary/individual graphs.
    """
    print(f"\n--- Starting Evaluation for '{run_name}' using DTW for ranking ---")
    netG.eval(); simulator_model.eval()

    dtw_results = []
    with torch.no_grad():
        for i in range(num_geometries):
            noise = torch.randn(1, latent_dim, 1, 1, device=device)
            generated_img = netG(noise, target_response.unsqueeze(0))
            predicted_response = simulator_model(resize_for_simulator(generated_img)).squeeze(0)

            target_np = target_response.cpu().numpy()
            pred_np = predicted_response.cpu().numpy()
            
            # Calculate DTW distance for ranking
            dtw_distance = dtw(pred_np, target_np, keep_internals=False).distance
            mae = np.mean(np.abs(pred_np - target_np))

            dtw_results.append({
                "image": generated_img.squeeze(0),
                "prediction": pred_np,
                "dtw_distance": dtw_distance,
                "mae": mae
            })

    # Sort results by DTW distance (lower is better)
    dtw_results.sort(key=lambda x: x['dtw_distance'])
    print("-> Generated and ranked geometries based on DTW distance.")

    # Create a dedicated subdirectory for this run
    run_output_dir = os.path.join(output_dir, f"dtw_evaluation_{run_name}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Save the ranked geometry images
    for rank, res in enumerate(dtw_results):
        img_path = os.path.join(run_output_dir, f"geom_rank_{rank+1}_dtw_{res['dtw_distance']:.4f}.png")
        vutils.save_image(res['image'], img_path, normalize=True)

    # --- Plot 1: Summary Graph ---
    target_np = target_response.cpu().numpy()
    all_predictions_np = np.array([res['prediction'] for res in dtw_results])
    avg_prediction = np.mean(all_predictions_np, axis=0)
    min_prediction = np.min(all_predictions_np, axis=0)
    max_prediction = np.max(all_predictions_np, axis=0)
    best_prediction = dtw_results[0]['prediction']
    angle_indices = np.arange(num_angles)
    x_labels = get_angle_labels(num_angles, is_radians=True)

    plt.figure(figsize=(12, 7))
    plt.plot(angle_indices, target_np, 'k-', linewidth=3, label='Target Response', zorder=5)
    plt.plot(angle_indices, best_prediction, 'g--', linewidth=2, label=f'Best Prediction (DTW: {dtw_results[0]["dtw_distance"]:.4f})', zorder=4)
    plt.plot(angle_indices, avg_prediction, 'r-', linewidth=2, label='Average Prediction', zorder=3)
    plt.fill_between(angle_indices, min_prediction, max_prediction, color='red', alpha=0.2, label='Min/Max Range')
    plt.xticks(ticks=angle_indices, labels=x_labels, rotation=45, ha="right")
    plt.title(f'Target vs. Generated Predictions for "{run_name}"', fontsize=16)
    plt.xlabel("Angle (radians)", fontsize=12)
    plt.ylabel("Absorbance", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir, "summary_comparison_graph.png"))
    plt.close()

    # --- Plot 2: Individual Graphs for each generated geometry ---
    for rank, res in enumerate(dtw_results):
        plt.figure(figsize=(8, 5))
        plt.plot(angle_indices, target_np, 'k-', linewidth=2, label='Target Response')
        plt.plot(angle_indices, res['prediction'], 'ro--', linewidth=2, label='Predicted Response')
        plt.xticks(ticks=angle_indices, labels=x_labels, rotation=45, ha="right")
        plt.title(f'"{run_name}" - Geometry Rank {rank+1} (DTW: {res["dtw_distance"]:.4f} | MAE: {res["mae"]:.4f})')
        plt.xlabel("Angle (radians)")
        plt.ylabel("Absorbance")
        plt.grid(True)
        plt.legend()
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(run_output_dir, f"individual_rank_{rank+1}.png"))
        plt.close()

    print(f"-> Saved ranked geometries and all graphs to '{run_output_dir}'")
    netG.train(); simulator_model.train()


def generate_full_dataset_comparison(netG, simulator_model, full_train_dataset: DiverseMetasurfaceDataset, device, output_dir,
                                       num_images_per_test, resize_for_sim_func, num_angles, latent_dim):
    """
    Generates a comparison plot for every single image in the training dataset by
    generating candidate images and finding the best match.
    """
    print(f"\n--- Generating Full Dataset Comparison Plots ---")
    netG.eval(); simulator_model.eval()

    comparison_plots_dir = os.path.join(output_dir, "full_dataset_comparison")
    os.makedirs(comparison_plots_dir, exist_ok=True)
    print(f"Saving {len(full_train_dataset)} comparison plots to: {comparison_plots_dir}")

    for i in range(len(full_train_dataset)):
        real_image, real_absorbance = full_train_dataset[i]
        img_filename = full_train_dataset.metadata_df.iloc[i][full_train_dataset.filename_col]
        img_title_base = os.path.splitext(os.path.basename(img_filename))[0]
        
        # Generate candidate images and find the best one
        generated_data = _generate_and_evaluate_images_conditional(
            netG, simulator_model, real_absorbance, real_absorbance, 
            num_images_per_test, device, latent_dim, resize_for_sim_func
        )
        
        output_plot_path = os.path.join(comparison_plots_dir, f"comparison_{img_title_base}.png")
        _plot_single_comparison(
            real_image_tensor=real_image,
            real_absorbance=real_absorbance,
            generated_data=generated_data,
            num_angles=num_angles,
            output_path=output_plot_path,
            img_title_base=img_title_base
        )
        
        if (i + 1) % 50 == 0:
            print(f"  ... generated {i+1}/{len(full_train_dataset)} comparison plots.")

    print("\nFinished generating all comparison graphs.")
    netG.train(); simulator_model.train()

# --- Helper functions for Conditional Mode ---

def _generate_and_evaluate_images_conditional(netG, simulator_model, real_absorbance, target_absorbance_for_gen, num_images, device, latent_dim, resize_for_sim_func):
    """Helper to generate images, evaluate them against the REAL absorbance, and sort by accuracy."""
    generated_data = []
    target_tensor = target_absorbance_for_gen.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(num_images):
            noise = torch.randn(1, latent_dim, 1, 1, device=device)
            generated_image = netG(noise, target_tensor).detach()
            pred_abs = simulator_model(resize_for_sim_func(generated_image)).detach()
            # Compare prediction to the REAL absorbance for accuracy calculation
            accuracy, rmse = calculate_absorbance_accuracy(real_absorbance.cpu(), pred_abs.cpu())
            generated_data.append((accuracy, rmse, generated_image, pred_abs.squeeze(0)))
    
    generated_data.sort(key=lambda x: x[0], reverse=True) # Sort by accuracy (desc)
    return generated_data

def _plot_single_comparison(real_image_tensor, real_absorbance, generated_data, num_angles, output_path, img_title_base):
    """Creates a single plot comparing one real image to its best generated counterpart."""
    if not generated_data:
        print(f"Warning: No generated data for {img_title_base}, skipping plot.")
        return

    best_gen_accuracy, best_gen_rmse, best_gen_img_tensor, best_gen_pred_abs = generated_data[0]
    
    real_image_display = (real_image_tensor * 0.5 + 0.5).clamp(0, 1).cpu().squeeze().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle(f"Detailed Comparison for '{img_title_base}'", fontsize=16)
    
    x_ticks = np.arange(num_angles)
    x_labels = get_angle_labels(num_angles)
    
    # Top-Left: Real Image
    axes[0, 0].imshow(real_image_display, cmap='gray')
    axes[0, 0].set_title("Real Image")
    axes[0, 0].axis('off')

    # Top-Right: Best Generated Image
    gen_img_display = (best_gen_img_tensor * 0.5 + 0.5).clamp(0, 1).cpu().squeeze().numpy()
    axes[0, 1].imshow(gen_img_display, cmap='gray')
    axes[0, 1].set_title(f"Best Generated Image (Acc: {best_gen_accuracy:.2f}%)")
    axes[0, 1].axis('off')

    # Bottom Plot (spanning both columns)
    ax_bottom = plt.subplot(2, 1, 2)
    ax_bottom.plot(x_ticks, real_absorbance.cpu().numpy(), label='Real Absorbance', marker='o', color='blue', alpha=0.7)
    ax_bottom.plot(x_ticks, best_gen_pred_abs.cpu().numpy(), label='Predicted Absorbance (from Gen. Img)', marker='x', linestyle='--', color='red')
    ax_bottom.set_title(f"Absorbance Comparison (RMSE: {best_gen_rmse:.4f})")
    ax_bottom.set_ylim(-0.1, 1.1); ax_bottom.legend(); ax_bottom.grid(True)
    ax_bottom.set_xlabel("Angle Index"); ax_bottom.set_ylabel("Absorbance")
    ax_bottom.set_xticks(x_ticks)
    ax_bottom.set_xticklabels(x_labels, rotation=45, ha="right")
    
    # Remove the original bottom axes
    fig.delaxes(axes[1,0])
    fig.delaxes(axes[1,1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)
