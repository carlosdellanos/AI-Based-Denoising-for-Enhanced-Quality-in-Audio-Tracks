# Class for adding quantization noise with dither.
# Parameters:
#   num_bits (int): Bit depth used for quantization.
#   dither_type (str): Type of dither applied before quantization.
#                      Options: 'uniform' or 'gaussian'.
#   dither_amplitude (float, optional): Amplitude of the dither noise.
#                                       If None, defaults to step_size / 8.
import torch

class AddQuantizationNoiseWithDither:
    def __init__(self, num_bits=8, dither_type='uniform', dither_amplitude=None):
        self.num_bits = num_bits
        self.dither_type = dither_type
        self.q_levels = 2 ** num_bits

        
        self.step_size = 2 / (self.q_levels - 1)                                                        # Signal range assumed to be [-1,1] -> 2 units
                                                                                                        # Quantization step size is then computed 
        if dither_amplitude is None:
            self.dither_amplitude = self.step_size / 8
        else:
            self.dither_amplitude = dither_amplitude

    def __call__(self, x, id=None, segment_index=None):
        x_clipped = torch.clamp(x, -1.0, 1.0)

        # Adding dither before quantization 
        if self.dither_type == 'uniform':
            dither = torch.empty_like(x_clipped).uniform_(-self.dither_amplitude, self.dither_amplitude)
        elif self.dither_type == 'gaussian':
            dither = torch.randn_like(x_clipped) * (self.dither_amplitude / 1.732)                      # Dither ampltude is scaled so that its energy
                                                                                                        # is similar to the uniform case
        else:
            raise ValueError(f"Dither type not supported: {self.dither_type}")

        x_dithered = x_clipped + dither
        x_dithered = torch.clamp(x_dithered, -1.0, 1.0)

        # Quantization
        x_scaled = (x_dithered + 1) * ((self.q_levels - 1) / 2)                                         # Map [-1,1] -> [0, q_levels-1]
        x_quantized = torch.round(x_scaled)
        x_dequantized = x_quantized / ((self.q_levels - 1) / 2) - 1

        return x_dequantized

