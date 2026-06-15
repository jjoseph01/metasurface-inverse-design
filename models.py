# models.py
import torch
import torch.nn as nn
import numpy as np


from config import (
   CHANNELS, LATENT_DIM, GF, DF, SIM_NDF, NUM_ANGLES,
   SIMULATOR_IMAGE_SIZE, CURRENT_GAN_IMAGE_SIZE, DROPOUT_RATE_SIM
)


class Generator(nn.Module):
   """
   The original Generator network, conditioned on an absorbance vector.
   This version outputs a grayscale image using Tanh, allowing for more complex patterns.
   """
   def __init__(self, target_image_size=CURRENT_GAN_IMAGE_SIZE, num_angles=NUM_ANGLES, latent_dim=LATENT_DIM, gf=GF, channels=CHANNELS):
       super(Generator, self).__init__()
       # Calculate the number of upsampling layers needed based on the target image size
       num_upsample_layers = int(np.log2(target_image_size / 4))
       initial_gf_multiplier = 2**(num_upsample_layers)


       # Define the network layers
       layers = [
           # Initial ConvTranspose layer from the latent vector + absorbance vector
           nn.ConvTranspose2d(latent_dim + num_angles, gf * initial_gf_multiplier, 4, 1, 0, bias=False),
           nn.BatchNorm2d(gf * initial_gf_multiplier),
           nn.ReLU(True)
       ]


       # Add the upsampling layers
       current_gf_multiplier = initial_gf_multiplier
       for i in range(num_upsample_layers - 1):
           layers.extend([
               nn.ConvTranspose2d(gf * current_gf_multiplier, gf * (current_gf_multiplier // 2), 4, 2, 1, bias=False),
               nn.BatchNorm2d(gf * (current_gf_multiplier // 2)),
               nn.ReLU(True)
           ])
           current_gf_multiplier //= 2
      
       # Final layer to produce the image
       layers.extend([
           nn.ConvTranspose2d(gf * current_gf_multiplier, channels, 4, 2, 1, bias=False),
           nn.Tanh() # Tanh activation produces grayscale images in the range [-1, 1]
       ])
      
       self.main = nn.Sequential(*layers)


   def forward(self, noise, absorbance_vector):
       # Reshape absorbance vector to be concatenated with noise
       expanded_absorbance = absorbance_vector.unsqueeze(-1).unsqueeze(-1)
       combined_input = torch.cat((noise, expanded_absorbance), dim=1)
       return self.main(combined_input)


# Conditional Critic for the main pipeline.
class ConditionalCritic(nn.Module):
   def __init__(self, target_image_size=CURRENT_GAN_IMAGE_SIZE, num_angles=NUM_ANGLES, df=DF, channels=CHANNELS):
       super(ConditionalCritic, self).__init__()
       num_downsample_layers = int(np.log2(target_image_size / 4)) + 1
       layers = [
           nn.Conv2d(channels + num_angles, df, 4, 2, 1, bias=False),
           nn.LeakyReLU(0.2, inplace=True)
       ]
       current_df_multiplier = 1
       for i in range(num_downsample_layers - 2):
           layers.extend([
               nn.Conv2d(df * current_df_multiplier, df * (current_df_multiplier * 2), 4, 2, 1, bias=False),
               nn.LeakyReLU(0.2, inplace=True)
           ])
           current_df_multiplier *= 2
       layers.append(nn.Conv2d(df * current_df_multiplier, 1, 4, 1, 0, bias=False))
       self.main = nn.Sequential(*layers)


   def forward(self, image_input, absorbance_vector):
       expanded_absorbance = absorbance_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, image_input.shape[2], image_input.shape[3])
       combined_input = torch.cat((image_input, expanded_absorbance), dim=1)
       return self.main(combined_input).view(-1, 1).squeeze(1)


# Unconditional Critic (kept for potential future use).
class UnconditionalCritic(nn.Module):
   def __init__(self, target_image_size=CURRENT_GAN_IMAGE_SIZE, df=DF, channels=CHANNELS):
       super(UnconditionalCritic, self).__init__()
       num_downsample_layers = int(np.log2(target_image_size / 4)) + 1
       layers = [
           nn.Conv2d(channels, df, 4, 2, 1, bias=False),
           nn.LeakyReLU(0.2, inplace=True)
       ]
       current_df_multiplier = 1
       for i in range(num_downsample_layers - 2):
           layers.extend([
               nn.Conv2d(df * current_df_multiplier, df * (current_df_multiplier * 2), 4, 2, 1, bias=False),
               nn.LeakyReLU(0.2, inplace=True)
           ])
           current_df_multiplier *= 2
       layers.append(nn.Conv2d(df * current_df_multiplier, 1, 4, 1, 0, bias=False))
       self.main = nn.Sequential(*layers)


   def forward(self, image_input):
       return self.main(image_input).view(-1, 1).squeeze(1)




# --- NEW SIMULATOR ARCHITECTURE from simulator4.ipynb ---


class ResidualBlock(nn.Module):
   """Residual block used in the simulator."""
   def __init__(self, in_channels, dropout_rate):
       super(ResidualBlock, self).__init__()
       self.block = nn.Sequential(
           nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
           nn.BatchNorm2d(in_channels),
           nn.ReLU(inplace=True),
           nn.Dropout2d(p=dropout_rate / 2),
           nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
           nn.BatchNorm2d(in_channels)
       )
   def forward(self, x):
       return x + self.block(x)


class RobustSimulatorCNN(nn.Module):
   """
   This is the deeper, more powerful version of the simulator CNN.
   It includes more convolutional and residual blocks to improve feature extraction
   for complex absorbance spectra"""
   def __init__(self, num_outputs=NUM_ANGLES, ndf=SIM_NDF, channels=CHANNELS, dropout_rate=DROPOUT_RATE_SIM):
       super(RobustSimulatorCNN, self).__init__()
      
       self.cnn_layers = nn.Sequential(
           # Input: (channels) x 64 x 64
           nn.Conv2d(channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
           nn.LeakyReLU(0.2, inplace=True),
           # State: (ndf) x 32 x 32


           nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
           nn.BatchNorm2d(ndf * 2),
           nn.LeakyReLU(0.2, inplace=True),
           ResidualBlock(ndf * 2, dropout_rate),
           # State: (ndf*2) x 16 x 16


           # --- NEW BLOCK ---
           nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
           nn.BatchNorm2d(ndf * 4),
           nn.LeakyReLU(0.2, inplace=True),
           ResidualBlock(ndf * 4, dropout_rate), # Added residual block
           ResidualBlock(ndf * 4, dropout_rate), # Added second residual block for more depth
           # State: (ndf*4) x 8 x 8


           nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
           nn.BatchNorm2d(ndf * 8),
           nn.LeakyReLU(0.2, inplace=True),
           # State: (ndf*8) x 4 x 4
          
           nn.AdaptiveAvgPool2d(1)
       )
      
       # The size of the vector after the CNN and pooling layers
       cnn_out_size = ndf * 8


       self.fc_layers = nn.Sequential(
           nn.Linear(cnn_out_size, 512), # Increased size of first FC layer
           nn.ReLU(True),
           nn.Dropout(dropout_rate),
           nn.Linear(512, 256),
           nn.ReLU(True),
           nn.Dropout(dropout_rate),
           nn.Linear(256, num_outputs),
           nn.Sigmoid()
       )


   def forward(self, image_input):
       cnn_out = self.cnn_layers(image_input)
       flattened = cnn_out.view(cnn_out.size(0), -1)
       output = self.fc_layers(flattened)
       return output
