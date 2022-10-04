import torch
from einops import einops
from torch import nn

"""
    Reference: Fast Transformer Decoding: One Write-Head is All You Need
    Reference: Attention is all you need
    Reference: Visual Transformer
"""

### Helpers and Functions

def DenseSiLU(x):
    return x

def silu(x):
    return nn.SiLU(x)

class SHA(nn.Module):
    def __init__(self):
        super(SHA, self).__init__()
    def forward(self, q, K, V):
        """ 
        Regular DPA.
        Params:
        m: Length of the input
        k: key dimension
        v: value dimension
        d: embedding dimension (d_model)
        h: number of heads
        Args:
        q: a vector with shape [k]
        K: a matrix with shape [m, k]
        V: a matrix with shape [m, v]
        Returns:
        y: a vector with shape [v]
        """

        logits = torch.einsum('k, mk -> m', K, q)
        key_weights = torch.nn.Softmax(logits)
        return torch.einsum('mv, m -> v', key_weights, V)

def linear_ema(x, alpha, delta):
    N, D, H = x.shape
    for i in range(D):
        for j in range(0, N - 1):
            x[j+1][i] = x[j][i] + x[j+1][i]
    
    return x

class EMALayer(nn.Module):
    def __init__(self, alpha=0.8, delta=0.1):
        super(EMALayer, self).__init__()
        D = 32
        H = 128

        # attention-specific params
        Z = 32
        V = 32

        self.H = 128
        self.alpha = torch.zeros((D, ))
        self.delta = torch.zeros((D, ))
        self.n = torch.zeros((D, H)) # lol n

        # projection matrices (make Linear later)
        self.beta = torch.zeros((D, H))
        self.mu = torch.zeros((D, H))

        # weights
        self.l_z  = torch.nn.Linear(N, Z) # weight to compute intermediate z
        self.l_v = torch.nn.Linear(N, V)

        # learnable scalars and offsets
        self.k_q = torch.nn.Parameter((Z,), requires_grad=True)
        self.mu_q = torch.nn.Parameter((Z,), requires_grad=True)
        self.k_k = torch.nn.Parameter((Z,), requires_grad=True)
        self.mu_k = torch.nn.Parameter((Z,), requires_grad=True)

        # attention operation    
        self.sha = SHA()

        # other
        self.b_rel = pos_bias(N)
    def forward(self, x):
        # N is time dimension
        # D is d_model
        # H is the dimension we are expanding into
        # we want a vector of shape N, D, H

        N, D = x.shape
        x_p = torch.einsum('ND, DH -> NDH', x, self.beta) #x @ self.beta
        x_i = linear_ema(x_p, self.alpha, self.delta)
        x_i = torch.einsum('NDH, DH -> ND')

        z = self.l_z(x_i)

        Q = z * k_q + mu_q
        K = z * k_k + mu_k
        V = self.l_v(x)

        O = self.sha(Q, K, V)

        
        z = l_z(x_e)
        x_o = x_e @ self.n

        return x_o

class MegaLayer(nn.Module):
    def __init__(self):
        super(MEGA, self).__init__()
        self.ema = EMA()
        self.sha = SHA()
        self.gate_y = nn.Identity()

    def forward(self):
        ema_out = self.ema(x)
        Q = ema_out
        K = ema_out
        V = x
        o = self.sha(Q, K, V)

class MegaBlock(nn.Module):
    def __init__(self):
        super(MEGA, self).__init__()

        input_shape = 32
        output_shape = 32

        self.megalayer = MegaLayer()
        self.norm_1 = Normalize()
        self.norm_2 = Normalize()
        self.ff = torch.nn.Linear(input_shape, output_shape)

    def forward(self):
        x = self.megalayer(x)
        x = self.norm_1(self)
        x_p = x
        x = self.linear(x)
        return x + x_p

def test_ema():
    H = 50 # dimension of ema
    N = 5 # batchsize
    D = 32 # d_model
    layer = EMALayer()
    i = torch.zeros((5, 32))
    layer(i)

if __name__ == "__main__":
    test_ema()
