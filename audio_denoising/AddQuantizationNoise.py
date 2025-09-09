# Class for adding quantization noise of a selected bit depth
# num_bits (int): Bit depth used for quantization. 
import torch

class AddQuantizationNoise:
    def __init__(self, num_bits=8):
        self.num_bits = num_bits

    def __call__(self, x, id = None, segment_index=None):
        x_clipped = torch.clamp(x, -1.0, 1.0)                   # Clip the signal to ensure it is in the desired range
        q_levels = 2 ** self.num_bits
        x_scaled = (x_clipped + 1) * (q_levels / 2 - 1)         # Map [-1,1] -> [0, q_levels-1]
        x_quantized = torch.round(x_scaled)                     # Quantize
        x_dequantized = x_quantized / (q_levels / 2 - 1) - 1    # Return to [-1,1] range
        return x_dequantized

