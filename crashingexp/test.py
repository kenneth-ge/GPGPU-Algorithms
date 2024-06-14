import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32

def ceildiv(a, b):
    return -(a // -b)

@triton.jit
def exp(Y, stride_yn, X, stride_xn, N):
    n = tl.program_id(0)
    BLOCK_SIZE: tl.constexpr = 1024
    n_block = tl.arange(0, BLOCK_SIZE)
    
    z = tl.load(X + n * stride_xn + n_block, mask=n_block < N)
    
    value = tl.exp(z)

    Y = Y + n * stride_yn + n_block
    tl.store(Y, value, mask=n_block < N)
    

import torch
# Allocate input/output tensors
# torch.zeros(512, 2, device='cuda')
X = torch.normal(0, 1, size=(1024,), device='cuda')
Y = torch.empty_like(X, device='cuda')

print(X.dtype)

# SPMD launch grid
grid = (ceildiv(X.shape[0], 1024),)

print(grid)

# enqueue GPU kernel
exp[grid](Y, Y.stride(0), 
              X, X.stride(0),
              X.shape[0])

torch_out = torch.exp(X)

print(Y)
print(torch_out)

print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(torch_out - Y))}')
