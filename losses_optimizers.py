# losses_optimizers.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from config import (
    INITIAL_LR_G, INITIAL_LR_C, LEARNING_RATE_SIM,
    BETA1_ADAM, BETA2_ADAM, DEVICE, WEIGHT_DECAY_SIMULATOR,
    LAMBDA_GRAD
)

class ShapeAndMagnitudeLoss(nn.Module):
    """
    Custom loss that combines point-wise magnitude loss (L1) with a
    gradient-based shape loss (L1) to better match the absorbance curve's shape.
    """
    def __init__(self, lambda_grad=0.5):
        super(ShapeAndMagnitudeLoss, self).__init__()
        self.lambda_grad = lambda_grad
        self.l1_loss = nn.L1Loss() # Use L1 for more interpretable point-wise error

    def forward(self, y_pred, y_true):
        # 1. Point-wise loss (for magnitude accuracy)
        loss_magnitude = self.l1_loss(y_pred, y_true)

        # 2. Gradient loss (for shape accuracy)
        pred_gradient = y_pred[:, 1:] - y_pred[:, :-1]
        true_gradient = y_true[:, 1:] - y_true[:, :-1]
        loss_shape = self.l1_loss(pred_gradient, true_gradient)

       # 3. Combine the losses
        combined_loss =  loss_magnitude + self.lambda_grad * loss_shape
        return combined_loss

def get_simulator_criterion():
    """Returns the new Shape and Magnitude loss function for the simulator."""
    return ShapeAndMagnitudeLoss(lambda_grad=LAMBDA_GRAD)

def get_optimizers_and_schedulers(netG, netC, simulator_model, decay_steps_total):
    """
    Sets up optimizers and learning rate schedulers for all models.
    """
    optimizerC = optim.Adam(netC.parameters(), lr=INITIAL_LR_C, betas=(BETA1_ADAM, BETA2_ADAM))
    optimizerG = optim.Adam(netG.parameters(), lr=INITIAL_LR_G, betas=(BETA1_ADAM, BETA2_ADAM))
    
    optimizer_sim = None
    if simulator_model is not None:
        optimizer_sim = optim.Adam(
            simulator_model.parameters(),
            lr=LEARNING_RATE_SIM,
            weight_decay=WEIGHT_DECAY_SIMULATOR
        )

    scheduler_G = CosineAnnealingLR(optimizerG, T_max=decay_steps_total, eta_min=INITIAL_LR_G*0.001)
    scheduler_C = CosineAnnealingLR(optimizerC, T_max=decay_steps_total, eta_min=INITIAL_LR_C*0.001)
    
    scheduler_sim = None
    if optimizer_sim is not None:
        # FIX APPLIED: Removed the 'verbose=True' argument which is deprecated in newer PyTorch versions.
        scheduler_sim = ReduceLROnPlateau(optimizer_sim, 'min', factor=0.2, patience=10)

    return optimizerG, optimizerC, optimizer_sim, scheduler_G, scheduler_C, scheduler_sim


def compute_gradient_penalty_conditional(critic_net, real_samples, fake_samples, real_absorbances, current_batch_size_gp):
    """Computes the gradient penalty for the conditional WGAN-GP."""
    alpha = torch.rand(current_batch_size_gp, 1, 1, 1, device=DEVICE)
    interpolates_img = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    c_interpolates = critic_net(interpolates_img, real_absorbances)
    
    gradients = torch.autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates_img,
        grad_outputs=torch.ones_like(c_interpolates, device=DEVICE),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty_val = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty_val

def compute_gradient_penalty_unconditional(critic_net, real_samples, fake_samples, current_batch_size_gp):
    """Computes the gradient penalty for the unconditional WGAN-GP."""
    alpha = torch.rand(current_batch_size_gp, 1, 1, 1, device=DEVICE)
    interpolates_img = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    c_interpolates = critic_net(interpolates_img)
    
    gradients = torch.autograd.grad(
        outputs=c_interpolates,
        inputs=interpolates_img,
        grad_outputs=torch.ones_like(c_interpolates, device=DEVICE),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty_val = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty_val
