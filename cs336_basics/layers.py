import torch
# We only import Module from torch.nn as permitted [cite: 19]
from torch.nn import Module 
from .nn import Linear 

class SwiGLU(Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        # d_ff is canonically 8/3 * d_model [cite: 649]
        # We ensure it's a multiple of 64 for hardware efficiency [cite: 659]
        self.d_ff = int((8/3 * d_model + 63) // 64 * 64)
        
        # SwiGLU requires three weight matrices: W1, W2, and W3 [cite: 649]
        self.W1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        # Formula: W2(SiLU(W1x) * W3x) [cite: 648]
        # SiLU is x * sigmoid(x). Using torch.sigmoid is permitted [cite: 658]
        w1_x = self.W1(x)
        gate = w1_x * torch.sigmoid(w1_x) 
        
        # Element-wise product with the W3 path [cite: 648]
        intermediate = gate * self.W3(x)
        
        return self.W2(intermediate)