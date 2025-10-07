import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.samplers import EulerSampler

class RFLoss(nn.Module):
    """
    Rectified-Flow Loss (velocity matching)
    """
    def __init__(
        self,
        target_channels: int,
        z_channels: int,
        depth: int,
        width: int,
        num_sampling_steps: int,
        grad_checkpointing: bool = False,
        cfg_dropout_p: float = 0.0,   # 可选：训练时做无条件dropout
        learn_null: bool = False,     # 可选：学一个null token
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_channels = target_channels
        self.num_sampling_steps = num_sampling_steps
        self.cfg_dropout_p = cfg_dropout_p

        self.learn_null = learn_null
        self.null_token = None
        if learn_null:
            self.null_token = nn.Parameter(torch.zeros(1, z_channels, device=device, dtype=dtype))

        self.device = device
        self.dtype = dtype

        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        ).to(device=device, dtype=dtype)

        self.rf_train = RectifiedFlow(
            data_shape=(target_channels,),
            velocity_field=lambda x_t, t, z: self.net(x_t, t, z),
            device=device,
            dtype=dtype,
        )

        self.rf_infer = RectifiedFlow(
            data_shape=(target_channels,),
            velocity_field=lambda x_t, t: x_t,  # placeholder, will be set in sample()
            device=device,
            dtype=dtype,
        )

    def forward(self, target, z, mask=None, return_patch_loss=False):
        if self.cfg_dropout_p > 0.0:
            with torch.no_grad():
                drop = (torch.rand(z.shape[0], device=z.device) < self.cfg_dropout_p).float().unsqueeze(1)
            if self.null_token is not None:
                z = torch.where(drop.bool().expand_as(z), self.null_token.expand_as(z), z)
            else:
                z = z * (1.0 - drop)

        t = self.rf_train.sample_train_time(target.shape[0])
        x_0 = torch.randn_like(target)
        x_t, dot_x_t = self.rf_train.get_interpolation(x_0=x_0, x_1=target, t=t)
        velocity = self.rf_train.get_velocity(x_t=x_t, t=t, z=z)
        loss = torch.mean((velocity - dot_x_t) ** 2, dim=1)

        patch_loss = loss.detach().clone()
        if mask is not None:
            loss = (loss * mask.view(loss.shape[0])).sum() / mask.sum()

        return (loss.mean(), patch_loss) if return_patch_loss else loss.mean()
    
    @torch.no_grad()
    def sample(
        self,
        z_cond,
        num_steps: int = None,
        cfg: float = 1.0,
    ):
        """
        B is batch * sequence length here.
        z_cond: (B, z_channels)
        z_uncond: (B, z_channels) 或 (1, z_channels). 若为 None 且 cfg!=1, 尝试用 null token 或全零。
        """
        num_steps = num_steps or self.num_sampling_steps
        B = z_cond.size(0)
        device, dtype = z_cond.device, z_cond.dtype

        x0 = torch.randn(B, self.in_channels, device=device, dtype=dtype)

        if cfg == 1.0:  # Turn off CFG
            def vfunc(x_t, t):
                return self.net(x_t, t, z_cond)
        else:  # Turn on CFG
            if self.null_token is not None:
                z_uncond = self.null_token.expand(1, -1)  # (1, d)
            else:
                z_uncond = torch.zeros(1, z_cond.size(1), device=device, dtype=dtype)

            if z_uncond.size(0) == 1:
                z_u = z_uncond.expand(B, -1)
            else:
                assert z_uncond.size(0) == B, "CFG z_uncond batch size must be 1 or match z_cond"
                z_u = z_uncond

            def vfunc(x_t, t):
                x_in = torch.cat([x_t, x_t], dim=0)
                if t.ndim == 0:
                    t_in = t.expand(2 * B)
                elif t.size(0) == B:
                    t_in = torch.cat([t, t], dim=0)
                else:
                    t_in = t  # 已经匹配
                c_in = torch.cat([z_u, z_cond], dim=0)
                v = self.net(x_in, t_in, c_in)  # (2B, C)
                v_u, v_c = v.chunk(2, dim=0)
                return v_u + cfg * (v_c - v_u)

        # 绑定到推理用 RF
        self.rf_infer.velocity_field = vfunc

        sampler = EulerSampler(rectified_flow=self.rf_infer)
        sampler.sample_loop(num_steps=num_steps, x_0=x0)
        x_1 = sampler.trajectories[-1]
        return x_1


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, t_scale=1000.0):
        t = t * t_scale
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)