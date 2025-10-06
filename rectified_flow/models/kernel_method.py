import torch
from torch import nn
from rectified_flow.flow_components.interpolation_solver import AffineInterp

class NadarayaWatson(nn.Module):
    def __init__(
        self, 
        pi_0_sample: torch.Tensor,
        pi_1_sample: torch.Tensor,
        sample_size: int,
        interp: AffineInterp = AffineInterp("straight"),
        bandwidth: float = 0.4,
        use_dot_x_t: bool = True,
    ):
        super().__init__()
        self.datashape = pi_0_sample.shape[1:]
        self.dataset_size = pi_0_sample.shape[0]
        self.pi_0_sample = pi_0_sample.detach().clone().reshape(pi_0_sample.shape[0], -1).requires_grad_(False)
        self.pi_1_sample = pi_1_sample.detach().clone().reshape(pi_1_sample.shape[0], -1).requires_grad_(False)
        assert self.pi_0_sample.shape == self.pi_1_sample.shape, "pi_0_sample and pi_1_sample must have the same shape"
        self.sample_size = sample_size
        self.interp = interp
        self.bandwidth = bandwidth
        self.use_dot_x_t = use_dot_x_t
        
    @torch.no_grad()
    def forward(self, z_t, t):
        """
        z_t: (batch_size, dim)
        t: (batch_size, )
        """
        idx = torch.randperm(self.dataset_size)[:self.sample_size]
        x_0 = self.pi_0_sample[idx, :].detach().clone() # (sample_size, dim)
        x_1 = self.pi_1_sample[idx, :].detach().clone() # (sample_size, dim)
        x_1 = x_1[None, :, :].expand(z_t.shape[0], -1, -1)     # (batch_size, sample_size, dim)
        x_0 = x_0[None, :, :].expand(z_t.shape[0], -1, -1)     # (batch_size, sample_size, dim)
        z_t = z_t[:, None, :].expand(-1, self.sample_size, -1) # (batch_size, sample_size, dim)
        
        a_t, b_t, dot_a_t, dot_b_t = self.interp.get_coeffs(t, detach=True) # (batch_size, )
        x_t = a_t[:, None, None] * x_1 + b_t[:, None, None] * x_0 # (batch_size, sample_size, dim)
        
        dzx = z_t - x_t
        logit = -torch.sum(dzx**2, dim=2) / (2. * self.bandwidth**2) # (batch_size, sample_size)
        
        prob = torch.softmax(logit, dim=1) # (batch_size, sample_size)

        if self.use_dot_x_t:
            # dot_a_t * x_1 + dot_b_t * x_0, (batch_size, sample_size, dim)
            vs = dot_a_t[:, None, None] * x_1 + dot_b_t[:, None, None] * x_0
        else:
            # equals to dot_x_t when z_t = x_t, (batch_size, sample_size, dim)
            vs = dot_a_t[:, None, None] * x_1 + (dot_b_t / b_t)[:, None, None] * (z_t - a_t[:, None, None] * x_1)
            
        v = torch.einsum("bn, bnd -> bd", prob, vs)
        
        return v.reshape(z_t.shape[0], *self.datashape)