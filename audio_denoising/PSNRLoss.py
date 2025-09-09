import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom loss function based on Peak Signal-to-Noise Ratio (PSNR).
# 
# Parameters:
#   max_val (float): Maximum possible value of the input signal (default: 80.0).
#   eps (float): Small constant to avoid division by zero (default: 1e-8).
#
# The loss computes the PSNR between the prediction and the target,
# and returns its negative value, so it can be minimized during training
# (since higher PSNR means better quality).

class PSNRLoss(nn.Module):
    def __init__(self, max_val=80.0, eps=1e-8):  # Prevents division by zero
        super().__init__()
        self.max_val = max_val
        self.eps = eps

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target, reduction='mean')
        psnr = 10 * torch.log10(self.max_val ** 2 / (mse + self.eps))
        return -psnr  # Negative because we want to minimize
