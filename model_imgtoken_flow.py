"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class ResidualBottleneckMLP(nn.Module):
    def __init__(self, emb_dim, r=4, dropout=0.0):
        super().__init__()
        hid = max(emb_dim // r, 1)
        self.norm = nn.LayerNorm(emb_dim)
        self.fc1  = nn.Linear(emb_dim, hid)
        self.fc2  = nn.Linear(hid, emb_dim)
        self.drop = nn.Dropout(dropout)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(self.fc2(h))
        return x + h

class DepthwiseSeparable(nn.Module):
    def __init__(self, C_in, C_out, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(C_in, C_in, kernel_size=3, stride=stride, padding=1, groups=C_in, bias=False)
        self.pw = nn.Conv2d(C_in, C_out, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, C_out) 
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        return self.act(x)

class VisionEncoderTiny(nn.Module):
    def __init__(self, in_channels, emb_dim=256, stem_channels=160,  # C 从 256 降到 160/128
                 use_dw_block=True, mlp_bottleneck_r=4, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.in_channels = in_channels

        self.patch = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=4, padding=3, bias=False),
            nn.GroupNorm(1, stem_channels),
            nn.GELU(),
        )

        self.block = DepthwiseSeparable(stem_channels, emb_dim, stride=2) if use_dw_block \
                     else nn.Conv2d(stem_channels, emb_dim, kernel_size=1, bias=False)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = ResidualBottleneckMLP(emb_dim, r=mlp_bottleneck_r, dropout=dropout)
        self.out_ln = nn.LayerNorm(emb_dim)

    @torch.compile(mode="reduce-overhead", fullgraph=False, dynamic=False)
    def _encode_flat(self, x):
        x = self.patch(x)     # -> (N, stemC, H/4, W/4)
        x = self.block(x)     # -> (N, emb,   H/8, W/8)  (若 use_dw_block)
        x = self.pool(x).flatten(1)   # (N, emb)
        x = self.head(x)
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 4:
            x = self._encode_flat(images)
            return self.out_ln(x)
        elif images.dim() == 5:
            B, T, C, H, W = images.shape
            x = images.reshape(B * T, C, H, W)
            x = self._encode_flat(x).reshape(B, T, self.emb_dim)
            return self.out_ln(x)
        else:
            raise ValueError(f"Expected 4D or 5D, got {tuple(images.shape)}")


class VisionEncoderFlattenMLPQiang(nn.Module):
    """
    Embed (B, T, C, H, W) images into (B, T, emb_dim) using a 2-layer MLP.
    The hidden width is chosen to match the parameter count of a V x emb_dim word embedding.
    """
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        emb_dim: int,
        vocab_size: int,
        hidden_width: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        P = in_channels * image_size * image_size  # flatten input dim
        E = emb_dim
        V = vocab_size

        if hidden_width is None:
            # Solve for H ~= (V*E - E) / (P + E + 1), then choose the closer of floor/ceil.
            H_real = (V * E - E) / (P + E + 1)
            H_floor = max(1, int(math.floor(H_real)))
            H_ceil  = max(1, int(math.ceil(H_real)))

            def mlp_params(H: int) -> int:
                return P*H + H + H*E + E  # include biases

            target = V * E  # word embedding params (no bias)
            pf, pc = mlp_params(H_floor), mlp_params(H_ceil)
            self.hidden_width = H_floor if abs(pf - target) <= abs(pc - target) else H_ceil
        else:
            self.hidden_width = int(hidden_width)
            assert self.hidden_width > 0

        H = self.hidden_width

        self.mlp = nn.Sequential(
            nn.Linear(P, H),
            nn.GELU(),
            nn.Linear(H, E),
        )

        self._init_weights()

        print(f"VisionEncoderFlattenMLPQiang has {self.param_count:,} params, "
              f"target {self.target_param_count:,} params, "
              f"hidden_width={self.hidden_width}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def target_param_count(self) -> int:
        return self.vocab_size * self.emb_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, T, C, H, W) or (B, C, H, W)
        returns: (B, T, emb_dim)
        """
        if images.dim() == 4:
            images = images.unsqueeze(1)
        if images.dim() != 5:
            raise ValueError(f"Expected (B,T,C,H,W) or (B,C,H,W), got {tuple(images.shape)}")

        B, T, C, H, W = images.shape
        if (C != self.in_channels) or (H != self.image_size) or (W != self.image_size):
            raise ValueError(
                f"Input (C,H,W)=({C},{H},{W}) does not match encoder config "
                f"({self.in_channels},{self.image_size},{self.image_size})."
            )

        x = images.reshape(B*T, C*H*W)   # flatten per token
        x = self.mlp(x)                  # (B*T, emb_dim)
        x = x.view(B, T, self.emb_dim)   # (B, T, emb_dim)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_vision_encoder: bool = True
    image_size: int = 32
    patch_size: int = 2
    in_channels: int = 1

    # add config for RF model
    use_rf_loss: bool = True
    rf_target_channels: int = 32 * 32
    rf_z_channels: int = 768  # condition
    rf_model_depth: int = 6
    rf_model_channel: int = 768
    rf_num_sampling_steps: int = 100
    rf_grad_checkpointing: bool = False
    rf_repeat: int = 4

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        modules = {}
        self._vision_tokens = None
        if config.use_vision_encoder:
            # vision_encoder = VisionEncoderTiny(
            #     in_channels=config.in_channels,
            #     emb_dim=config.n_embd,
            #     dropout=config.dropout,
            # )
            vision_encoder = VisionEncoderFlattenMLPQiang(
                in_channels=config.in_channels,
                image_size=config.image_size,
                emb_dim=config.n_embd,
                vocab_size=config.vocab_size,
            )
            modules['vte'] = vision_encoder
        else:
            modules['wte'] = nn.Embedding(config.vocab_size, config.n_embd)

        modules.update(dict(
            wpe = nn.Embedding(config.block_size, config.n_embd),
            flow_pe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.transformer = nn.ModuleDict(modules)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        if config.use_rf_loss:
            from models.RFloss import RFLoss
            self.rf_loss_model = RFLoss(
                target_channels=config.rf_target_channels,
                z_channels=config.rf_z_channels,
                depth=config.rf_model_depth,
                width=config.rf_model_channel,
                num_sampling_steps=config.rf_num_sampling_steps,
                grad_checkpointing=config.rf_grad_checkpointing,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
        else:
            self.rf_loss_model = None

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_gpt_encoder(self, imgs):
        if self.config.use_vision_encoder:
            encoder = self.transformer.vte
            if imgs.dim() != 5:
                raise ValueError(f"Vision encoder expects inputs of shape (B, T, C, H, W), got {tuple(imgs.shape)}")
            # weight = encoder.patch[0].weight
            weight = encoder.mlp[0].weight
            imgs = imgs.to(device=weight.device, dtype=weight.dtype)
            tok_emb = encoder(imgs) # (b, num_patches, n_embd)
        else:
            # if imgs.dim() != 2:
            #     raise ValueError(f"GPT expects token indices of shape (B, T), got {tuple(imgs.shape)}")
            # tok_emb = self.transformer.wte(imgs) # token embeddings of shape (b, t, n_embd)
            raise ValueError("GPT model without vision encoder is not supported here.")

        b, t, _ = tok_emb.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        if self._vision_tokens is not None and t != self._vision_tokens:
            raise ValueError(f"Vision encoder produced {t} tokens, expected {self._vision_tokens}")
        
        device = tok_emb.device
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb) # (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # (b, t, n_embd), the condition z for RF

        flow_pe = self.transformer.flow_pe(pos) # (t, n_embd)
        x = x + flow_pe
        return x

    def forward(self, idx, targets=None):
        x = self.forward_gpt_encoder(idx)
        B, T , _ = x.size()
        if targets is not None and self.config.use_rf_loss:
            target_images_flat = targets.view(B*T, -1).repeat(self.config.rf_repeat, 1)  # (B*T*repeat, C*H*W), repeat
            z = x.view(B*T, -1).repeat(self.config.rf_repeat, 1)
            loss = self.rf_loss_model(target_images_flat, z)
            logits = None
        elif targets is not None:
            # if we are given some desired targets also calculate the loss
            # logits = self.lm_head(x)
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            raise ValueError("GPT model without RF loss is not supported here.")
        else:
            raise ValueError("targets must be provided when using RF loss.")

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, image_bank=None, return_images=False, cfg=4.0):
        import time, pathlib, torch
        from torchvision.utils import save_image, make_grid

        def _to01(x):  # [-1,1] -> [0,1]
            return (x.clamp(-1, 1) + 1) * 0.5

        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        save_dir = f"/scratch/10992/liaorunlong93/logs/samples/chinese-image-flowhead-sample{time_str}"
        base = pathlib.Path(save_dir)
        (base / "by_step").mkdir(parents=True, exist_ok=True)

        if not (self.config.use_vision_encoder and self.config.use_rf_loss):
            raise ValueError("Only vision-conditioned generation with RF loss is supported here.")
        assert idx.dim() == 5, f"Vision generate expects (B,T,C,H,W), got {tuple(idx.shape)}"
        assert image_bank is not None, "image_bank (id->image table) is required when use_vision_encoder=True"

        try:
            ref_param = self.transformer.vte.mlp[0].weight
        except Exception:
            ref_param = next(self.transformer.parameters())
        device_ref, dtype_ref = ref_param.device, ref_param.dtype

        id_seq = None

        bank_imgs = image_bank.to(device=device_ref, dtype=dtype_ref, non_blocking=True)  # [V,C,H,W]
        bank_flat = bank_imgs.flatten(1).to(dtype=torch.float32)                          # [V,P]
        bank_b2 = (bank_flat ** 2).sum(dim=1, keepdim=True).t() 

        assert idx.dim() == 5, f"Vision generate expects (B,T,C,H,W), got {tuple(idx.shape)}"
        assert image_bank is not None, "image_bank (id->image table) is required when use_vision_encoder=True"

        # weight = self.transformer.vte.patch[0].weight
        weight = self.transformer.vte.mlp[0].weight
        img_seq = idx.to(device=weight.device, dtype=weight.dtype)  # [B, T0, C, H, W]
        print(f"Prompt image sequence shape: {tuple(img_seq.shape)}")
        
        B, T0, C, H, W = img_seq.shape
        P = C * H * W
        assert self.config.rf_target_channels == P, \
            f"rf_target_channels={self.config.rf_target_channels} != C*H*W={P}"

        steps = 200
        for t_new in range(max_new_tokens):
            print(f"seq length: {img_seq.size(1)} -> {img_seq.size(1)+1}")
            img_cond = img_seq if img_seq.size(1) <= self.config.block_size else img_seq[:, -self.config.block_size:]

            x = self.forward_gpt_encoder(img_cond)  # [B,T,n_embd]
            print(f"feature shape: {tuple(x.shape)}")
            z_last = x[:, -1, :]                    # [B,n_embd]
            z_last_rf = z_last.to(device=self.rf_loss_model.device, dtype=self.rf_loss_model.dtype, non_blocking=True)

            new_flat = self.rf_loss_model.sample(z_last_rf, num_steps=steps, cfg=cfg)  # [B,P]
            new_imgs = (
                new_flat.view(B, C, H, W)
                .to(device=img_seq.device, dtype=img_seq.dtype, non_blocking=True)
                .clamp_(-1.0, 1.0)
            )

            q = new_imgs.reshape(B, -1).to(dtype=torch.float32)     # [B,P]
            a2 = (q ** 2).sum(dim=1, keepdim=True)                  # [B,1]
            ab = q @ bank_flat.t()                                  # [B,V]
            d2 = a2 + bank_b2 - 2.0 * ab                            # [B,V]
            nn_id = torch.argmin(d2, dim=1, keepdim=True)           # [B,1], dtype=long

            id_seq = nn_id if id_seq is None else torch.cat([id_seq, nn_id], dim=1)

            # --- 保存成对对比图（相同前缀） ---
            # 单图：bXX_tYYY_gen.png & bXX_tYYY_nn.png
            gen01 = _to01(new_imgs.detach().cpu())
            nn_imgs = bank_imgs[nn_id.squeeze(1)].to(device="cpu", non_blocking=True)  # [B,C,H,W]
            nn01 = _to01(nn_imgs)

            for b in range(B):
                prefix = base / f"b{b:02d}_t{t_new:03d}"
                save_image(gen01[b], str(prefix) + "_gen.png")
                save_image(nn01[b], str(prefix) + "_nn.png")

            # --- 用最近邻作为下一 token 的“图片输入” ---
            next_imgs = bank_imgs[nn_id.squeeze(1)].to(device=img_seq.device, dtype=img_seq.dtype, non_blocking=True)
            img_seq = torch.cat([img_seq, next_imgs.unsqueeze(1)], dim=1)

        if return_images:
            return img_seq
        else:
            return id_seq

