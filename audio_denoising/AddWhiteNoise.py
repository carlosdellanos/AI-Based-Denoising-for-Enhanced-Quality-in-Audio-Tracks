# Class used for adding white Gaussian noise to the original clean signal
#
# Parameters:
#   std_noise (float): Standard deviation of the Gaussian noise to be added (default: 0.01)
#   train (bool): If True, noise is generated randomly; if False, noise is generated deterministically
#                 based on the input id for reproducibility
#
# During training, the class adds random white Gaussian noise.
# During evaluation/testing, the noise is deterministically generated using the provided id 
# to ensure consistency across runs.

import torch

class AddWhiteNoise:
    def __init__(self, std_noise=0.01, train=True):
        self.std = std_noise
        self.train = train

    def __call__(self, x, id=None, segment_index=None):
        if self.train:
            noise = torch.randn_like(x) * self.std
        else:
            if id is None:
                raise ValueError("An ID must be provided in eval mode for deterministic noise.")
            g = torch.Generator()
            g.manual_seed(id)
            noise = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype) * self.std

        return x + noise
