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
from pyexpat import model

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
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Causal mask
            # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
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
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if not config.use_vision_encoder:
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

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

    def forward(self, idx, targets=None):
        if self.config.use_vision_encoder:
            encoder = self.transformer.vte
            if idx.dim() != 5:
                raise ValueError(f"Vision encoder expects inputs of shape (B, T, C, H, W), got {tuple(idx.shape)}")
            # weight = encoder.patch[0].weight
            weight = encoder.mlp[0].weight
            idx = idx.to(device=weight.device, dtype=weight.dtype)
            tok_emb = encoder(idx) # (b, num_patches, n_embd)
        else:
            if idx.dim() != 2:
                raise ValueError(f"GPT expects token indices of shape (B, T), got {tuple(idx.shape)}")
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        b, t, _ = tok_emb.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        if self._vision_tokens is not None and t != self._vision_tokens:
            raise ValueError(f"Vision encoder produced {t} tokens, expected {self._vision_tokens}")
        
        device = tok_emb.device
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if (targets is not None) or self.config.use_vision_encoder:
            logits = self.lm_head(x)  # [B, T, V]
            loss = None if targets is None else F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x)
            loss = None

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
    def generate(
        self,
        batch_size,
        image_bank=None,
        N_steps: int = 50,
        t_mode: str = "async",     # "sync" / "async" / "sde" / "sde-async"
        top_k: int | None = None,
        verbose: bool = True,
        # === 新增：prompt 注入相关 ===
        prompt_ids: torch.Tensor | None = None,  # 长度为 n_prompt 的索引；可为 [n_prompt] 或 [B, n_prompt]
        freeze_prompt: bool = True,              # 若 True，则采样过程中始终不更新前 n_prompt 个 token
    ):
        """
        Args (仅列出和本改动相关的关键点):
            image_bank: [V, C, H, W]
            prompt_ids: 1D [n_prompt]（对 batch 复用）或 2D [B, n_prompt]（逐样本不同）
            freeze_prompt: 是否在采样循环中跳过对前 n_prompt 的更新
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        B = batch_size
        C = self.config.in_channels
        H = self.config.image_size
        W = self.config.image_size
        T = self.config.block_size

        assert image_bank is not None, "image_bank is required for vision-conditioned generation."
        image_bank = image_bank.to(device=device, dtype=torch.float32)  # [V, C, H, W]

        # 初始化为高斯
        x_t = torch.randn((B, T, C, H, W), device=device, dtype=torch.float32)

        # -------- 新增：将前 n_prompt 个 token 用 image_bank 中指定索引的图像注入 ----------
        n_prompt = 0
        if prompt_ids is not None:
            # 允许 [n_prompt] 或 [B, n_prompt]
            prompt_ids = prompt_ids.to(device='cpu', dtype=torch.long)
            if prompt_ids.dim() == 1:
                prompt_ids = prompt_ids.unsqueeze(0).expand(B, -1).contiguous()  # [B, n_prompt]
            elif prompt_ids.dim() == 2:
                assert prompt_ids.size(0) == B, f"prompt_ids first dim must be batch_size={B}"
            else:
                raise ValueError("prompt_ids must be 1D or 2D")

            n_prompt = prompt_ids.size(1)
            if n_prompt > T:
                raise ValueError(f"n_prompt={n_prompt} exceeds block_size T={T}")
            # 从 image_bank 取出对应图像并写入到前 n_prompt 个 token
            prompt_imgs = image_bank[prompt_ids]  # [B, n_prompt, C, H, W]
            x_t[:, :n_prompt] = prompt_imgs.to(x_t.device, x_t.dtype)
        # ----------------------------------------------------------------------

        def softmax_topk(logits: torch.Tensor, k: int | None, dim: int = -1) -> torch.Tensor:
            if (k is None) or (k <= 0) or (k >= logits.size(dim)):
                return torch.softmax(logits, dim=dim)
            topk_vals, topk_idx = torch.topk(logits, k, dim=dim)
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter(dim, topk_idx, True)
            masked_logits = logits.masked_fill(~mask, float('-inf'))
            return torch.softmax(masked_logits, dim=dim)

        mode = t_mode.lower()

        if mode == "sync":
            # 同步 ODE
            for i in range(N_steps):
                t = i / N_steps  # in [0,1)
                if verbose:
                    print(f"[sync] step {i+1}/{N_steps}, t={t:.5f}")
                logits, _ = self(x_t)                 # [B, T, V]
                p_t = softmax_topk(logits, top_k, -1) # [B, T, V]
                x_1_hat = torch.einsum('btv, vchw -> btchw', p_t, image_bank)  # [B, T, C, H, W]
                delta = (x_1_hat - x_t) / (1.0 - t) / N_steps
                if freeze_prompt and n_prompt > 0:
                    delta[:, :n_prompt] = 0
                x_t = x_t + delta

        elif mode in ("async", "ar"):
            # 逐 token ODE（自回归式推进）
            start_j = n_prompt if freeze_prompt else 0
            for j in range(start_j, T):
                for i in range(N_steps):
                    t = i / N_steps
                    if verbose:
                        print(f"[async] token {j+1}/{T}, step {i+1}/{N_steps}, t={t:.5f}")
                    logits, _ = self(x_t)          # [B, T, V]
                    logits_j = logits[:, j, :]     # [B, V]
                    p_t_j = softmax_topk(logits_j, top_k, -1)        # [B, V]
                    x1_hat_j = torch.einsum('bv, vchw -> bchw', p_t_j, image_bank)  # [B, C, H, W]
                    xj = x_t[:, j, :, :, :]
                    xj = xj + (x1_hat_j - xj) / (1.0 - t) / N_steps
                    x_t[:, j, :, :, :] = xj

        elif mode in ("sde", "diffusion", "sde-sync"):
            # 同步 SDE
            for i in range(N_steps):
                t_next = (i + 1) / N_steps
                if verbose:
                    print(f"[sde] step {i+1}/{N_steps}, t_next={t_next:.5f}")
                logits, _ = self(x_t)                 # [B, T, V]
                p_t = softmax_topk(logits, top_k, -1) # [B, T, V]
                x_1_hat = torch.einsum('btv, vchw -> btchw', p_t, image_bank)  # [B, T, C, H, W]
                eps = torch.randn_like(x_t)
                x_next = t_next * x_1_hat + (1.0 - t_next) * eps
                if freeze_prompt and n_prompt > 0:
                    x_next[:, :n_prompt] = x_t[:, :n_prompt]  # 保持前 n_prompt 不变
                x_t = x_next

        elif mode in ("sde-async", "sde_ar", "sde-ar"):
            # 逐 token SDE（自回归式推进）
            start_j = n_prompt if freeze_prompt else 0
            for j in range(start_j, T):
                for i in range(N_steps):
                    t_next = (i + 1) / N_steps
                    if verbose:
                        print(f"[sde-async] token {j+1}/{T}, step {i+1}/{N_steps}, t_next={t_next:.5f}")
                    logits, _ = self(x_t)          # [B, T, V]
                    logits_j = logits[:, j, :]     # [B, V]
                    p_t_j = softmax_topk(logits_j, top_k, -1)                      # [B, V]
                    x1_hat_j = torch.einsum('bv, vchw -> bchw', p_t_j, image_bank)  # [B, C, H, W]
                    eps_j = torch.randn_like(x_t[:, j, :, :, :])
                    x_t[:, j, :, :, :] = t_next * x1_hat_j + (1.0 - t_next) * eps_j

        else:
            raise ValueError(f"Unknown t_mode='{t_mode}', expected 'sync', 'async', 'sde', or 'sde-async'.")

        return x_t

