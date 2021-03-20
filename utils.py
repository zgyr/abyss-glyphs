import torch
import numpy as np

def get_map(size):
    x = torch.ones(size, size) * torch.linspace(-1, 1, steps=size)
    y = x.T
    r = torch.sqrt(x**2 + y**2) / (-2**0.5) + 1
    phi = torch.atan2(x, y)
    alpha = torch.sin(phi)
    beta = torch.cos(phi)
    return torch.stack([x, y, r, alpha, beta], dim=0).unsqueeze(0)

def img2sa(x):
    fft = torch.fft.fftshift(torch.fft.fft2(x))
    magnitude = torch.log2(torch.sqrt(fft.real**2 + fft.imag**2)) / 16
    phase = torch.atan2(fft.real, fft.imag)
    return torch.cat([magnitude, phase / 3.14159265], 1)

def sa2img(x):
    size = np.asarray(x.shape[-2:]) // 2
    magnitude, phase = x[:, 0:1], x[:, 1:2] * 3.14159265
    magnitude = 2**(magnitude * 16)
    result = torch.stack([magnitude * torch.sin(phase),
                          magnitude * torch.cos(phase)], -1)
    return torch.fft.ifft2(torch.view_as_complex(result)).abs()