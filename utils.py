# utils.py
# A utility file for common functions like weight initialization.
import torch.nn as nn

def weights_init(m):
    """
    Initializes weights of convolutional, batch normalization, and linear layers
    with a normal distribution (mean=0.0, std=0.02) and biases to zero.
    This is a common practice in DCGANs and WGANs.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
