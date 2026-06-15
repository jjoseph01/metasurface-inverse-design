# training_loops.py
import torch
import time
import os
import torchvision.utils as vutils
from collections import defaultdict
import numpy as np

from config import LATENT_DIM, BATCH_SIZE
from losses_optimizers import compute_gradient_penalty_conditional, compute_gradient_penalty_unconditional

def pretrain_simulator(simulator_model, train_dataloader, test_dataloader,
                       optimizer_sim, scheduler_sim, criterion_S, num_epochs, device,
                       output_dir):
    """Pre-trains the simulator model and saves the best performing version."""
    print("--- Starting Simulator Pre-training ---")
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    best_model_epoch = 0
    best_model_path = os.path.join(output_dir, 'best_simulator_model.pth')

    for epoch in range(num_epochs):
        simulator_model.train()
        running_train_loss = 0.0
        epoch_start_time = time.time()

        for images, targets in train_dataloader:
            images, targets = images.to(device), targets.to(device)
            optimizer_sim.zero_grad()
            outputs = simulator_model(images)
            loss = criterion_S(outputs, targets)
            loss.backward()
            optimizer_sim.step()
            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_train_loss)

        simulator_model.eval()
        running_test_loss = 0.0
        if test_dataloader:
            with torch.no_grad():
                for images_test, targets_test in test_dataloader:
                    images_test, targets_test = images_test.to(device), targets_test.to(device)
                    outputs_test = simulator_model(images_test)
                    loss_test = criterion_S(outputs_test, targets_test)
                    running_test_loss += loss_test.item() * images_test.size(0)
            
            epoch_test_loss = running_test_loss / len(test_dataloader.dataset)
            test_losses.append(epoch_test_loss)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}, Test Loss: {epoch_test_loss:.6f}, Time: {time.time()-epoch_start_time:.2f}s")
            
            if epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss
                best_model_epoch = epoch + 1
                torch.save({
                    'epoch': best_model_epoch,
                    'model_state_dict': simulator_model.state_dict(),
                    'optimizer_state_dict': optimizer_sim.state_dict(),
                    'best_test_loss': best_test_loss,
                }, best_model_path)
                print(f"    -> New best model saved to {best_model_path}")
            
            scheduler_sim.step(epoch_test_loss)
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}, Time: {time.time()-epoch_start_time:.2f}s")

    print("\nSimulator Training Finished.")
    if os.path.exists(best_model_path):
        print(f"Best model was saved from epoch {best_model_epoch} with Test Loss: {best_test_loss:.6f}")
        checkpoint = torch.load(best_model_path)
        simulator_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best simulator model from: {best_model_path}")
    else:
        print("Simulator training finished. Using the final model state.")
        
    return train_losses, test_losses

def train_gan_conditional(netG, netC, simulator_model, train_dataloader,
                          optimizerG, optimizerC, scheduler_G, scheduler_C,
                          criterion_S, num_epochs_gan, n_critic, gp_weight, lambda_sim_loss,
                          device, target_dirs, fixed_noise, fixed_absorbance_for_gen,
                          num_batches_per_epoch, resize_for_simulator_func):
    """Conducts the conditional GAN training."""
    G_losses, C_losses, Sim_losses_G_side = [], [], []
    logged_lr_g, logged_lr_c = [], []
    iters = 0

    checkpoints_dir = target_dirs["checkpoints"]
    training_images_dir = target_dirs["training_images"]

    print("--- Starting Conditional GAN Training Loop... ---")
    for epoch in range(num_epochs_gan):
        start_time_epoch = time.time()
        for i, (real_images, target_absorbances) in enumerate(train_dataloader):
            current_batch_s = real_images.size(0)
            real_images, target_absorbances = real_images.to(device), target_absorbances.to(device)
            for _ in range(n_critic):
                netC.zero_grad()
                noise = torch.randn(current_batch_s, LATENT_DIM, 1, 1, device=device)
                fake_images = netG(noise, target_absorbances).detach()
                real_scores = netC(real_images, target_absorbances)
                fake_scores_c = netC(fake_images, target_absorbances)
                gradient_penalty_val = compute_gradient_penalty_conditional(netC, real_images.data, fake_images.data, target_absorbances.data, current_batch_s)
                loss_c = fake_scores_c.mean() - real_scores.mean() + gp_weight * gradient_penalty_val
                loss_c.backward()
                optimizerC.step()
            scheduler_C.step(); logged_lr_c.append(optimizerC.param_groups[0]["lr"])
            netG.zero_grad()
            gen_noise = torch.randn(current_batch_s, LATENT_DIM, 1, 1, device=device)
            gen_fake_images = netG(gen_noise, target_absorbances)
            fake_scores_g = netC(gen_fake_images, target_absorbances)
            errG_C = -fake_scores_g.mean()
            gen_fake_images_for_sim = resize_for_simulator_func(gen_fake_images)
            simulator_output = simulator_model(gen_fake_images_for_sim)
            errG_S = criterion_S(simulator_output, target_absorbances)
            errG = errG_C + lambda_sim_loss * errG_S
            errG.backward()
            optimizerG.step()
            scheduler_G.step(); logged_lr_g.append(optimizerG.param_groups[0]["lr"])
            if iters % 50 == 0: print(f'[{epoch+1}/{num_epochs_gan}][{i}/{num_batches_per_epoch}] Loss_C: {loss_c.item():.4f} Loss_G: {errG.item():.4f}')
            G_losses.append(errG.item()); C_losses.append(loss_c.item()); Sim_losses_G_side.append(errG_S.item())
            iters += 1

        netG.eval()
        with torch.no_grad():
            fake_fixed_noise_images = netG(fixed_noise, fixed_absorbance_for_gen).detach().cpu()
            vutils.save_image(fake_fixed_noise_images, os.path.join(training_images_dir, f'fake_samples_epoch_{epoch+1:03d}.png'), normalize=True)
        netG.train()
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs_gan:
            torch.save({'generator_state_dict': netG.state_dict()}, os.path.join(checkpoints_dir, f'gan_ckpt_epoch_{epoch+1}.pth'))
            print(f"Saved GAN checkpoint to {checkpoints_dir}")
        print(f"Epoch {epoch+1} finished in {time.time() - start_time_epoch:.2f}s.")
    return G_losses, C_losses, Sim_losses_G_side, logged_lr_g, logged_lr_c

def train_gan_constant_target(netG, netC, simulator_model, gan_unconditional_loader,
                              optimizerG, optimizerC, scheduler_G, scheduler_C,
                              criterion_S, num_epochs_gan, n_critic, gp_weight, lambda_sim_loss,
                              device, target_dirs, fixed_noise, desired_response,
                              resize_for_simulator_func):
    """Conducts the GAN training targeting a single, constant spectral response."""
    G_losses, C_losses, Sim_losses_G_side = [], [], []
    logged_lr_g, logged_lr_c = [], []
    iters = 0

    checkpoints_dir = target_dirs["checkpoints"]
    training_images_dir = target_dirs["training_images"]

    print("--- Starting Constant-Target GAN Training Loop... ---")
    for epoch in range(num_epochs_gan):
        start_time_epoch = time.time()
        for i, real_unconditional_images in enumerate(gan_unconditional_loader):
            real_images_for_critic = real_unconditional_images.to(device)
            current_batch_s = real_images_for_critic.size(0)
            if current_batch_s != BATCH_SIZE: continue
            desired_response_batch = desired_response.repeat(current_batch_s, 1)
            for _ in range(n_critic):
                netC.zero_grad()
                noise = torch.randn(current_batch_s, LATENT_DIM, 1, 1, device=device)
                with torch.no_grad():
                    fake_images = netG(noise, desired_response_batch).detach()
                real_scores = netC(real_images_for_critic)
                fake_scores_c = netC(fake_images)
                gradient_penalty_val = compute_gradient_penalty_unconditional(netC, real_images_for_critic.data, fake_images.data, current_batch_s)
                loss_c = fake_scores_c.mean() - real_scores.mean() + gp_weight * gradient_penalty_val
                loss_c.backward()
                optimizerC.step()
            scheduler_C.step(); logged_lr_c.append(optimizerC.param_groups[0]["lr"])
            netG.zero_grad()
            gen_noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1, device=device)
            desired_response_batch_g = desired_response.repeat(BATCH_SIZE, 1)
            gen_fake_images = netG(gen_noise, desired_response_batch_g)
            fake_scores_g = netC(gen_fake_images)
            errG_C = -fake_scores_g.mean()
            gen_fake_images_for_sim = resize_for_simulator_func(gen_fake_images)
            simulator_output = simulator_model(gen_fake_images_for_sim)
            errG_S = criterion_S(simulator_output, desired_response_batch_g)
            errG = errG_C + lambda_sim_loss * errG_S
            errG.backward()
            optimizerG.step()
            scheduler_G.step(); logged_lr_g.append(optimizerG.param_groups[0]["lr"])
            if i % 50 == 0:
                print(f'[{epoch+1}/{num_epochs_gan}][{i}/{len(gan_unconditional_loader)}] Loss_C: {loss_c.item():.4f} Loss_G: {errG.item():.4f} C(real): {real_scores.mean().item():.4f} C(fake): {fake_scores_c.mean().item():.4f}')
            G_losses.append(errG.item()); C_losses.append(loss_c.item()); Sim_losses_G_side.append(errG_S.item())
            iters += 1

        netG.eval()
        with torch.no_grad():
            num_fixed_images = min(64, BATCH_SIZE)
            fixed_images = netG(fixed_noise[:num_fixed_images], desired_response.repeat(num_fixed_images, 1)).detach().cpu()
            vutils.save_image(fixed_images, os.path.join(training_images_dir, f'fake_samples_epoch_{epoch+1:03d}.png'), normalize=True)
        netG.train()
        if (epoch + 1) % 50 == 0 or (epoch + 1) == num_epochs_gan:
            torch.save({'generator_state_dict': netG.state_dict()}, os.path.join(checkpoints_dir, f'gan_ckpt_epoch_{epoch+1}.pth'))
            print(f"--- Saved GAN checkpoint to {checkpoints_dir} ---")
        print(f"--- Epoch {epoch+1} finished in {time.time() - start_time_epoch:.2f}s ---")
    return G_losses, C_losses, Sim_losses_G_side, logged_lr_g, logged_lr_c
