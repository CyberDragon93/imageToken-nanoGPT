"""
Accelerate-style training script for nanoGPT:
- Distributed training via 🤗 Accelerate (no manual DDP)
- Mixed precision via Accelerate (bf16/fp16) + autocast + scaler
- torch.compile for speed (PyTorch 2.0+)

Examples:
  Single GPU (debug):
    accelerate launch train.py --batch_size=32 --compile=False

  Single node, 4 GPUs:
    accelerate launch --num_processes=4 train.py

  2 nodes, 4 GPUs each (total 8 procs):
    # master
    accelerate launch --num_machines=2 --num_processes=4 --machine_rank=0 \
      --main_process_ip=123.456.123.456 --main_process_port=1234 train.py
    # worker
    accelerate launch --num_machines=2 --num_processes=4 --machine_rank=1 \
      --main_process_ip=123.456.123.456 --main_process_port=1234 train.py

(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

import glob
from PIL import Image
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import set_seed

from model_tokenfree_flow import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# -----------------------------------------------------------------------------
# I/O
date = time.strftime('%Y-%m-%d-%H-%M-%S')
out_dir = '/scratch/10992/liaorunlong93/logs/nanoGPT_CHN_3G/chinese-image-flowhead-' + date
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'nanoGPT_CHN_3G'
wandb_run_name = 'img_tokenizerf-flow_head' # 'run' + str(time.time())
# data
dataset = 'chinese_char'
batch_size = 64 # micro-batch size per *optimizer step loop*, before GAS
gradient_accumulation_steps = 4  # intended global GAS; will be adjusted by world_size below
eval_interval = 2000 // gradient_accumulation_steps
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# RF Loss 
use_rf_loss = True
rf_target_channels = 1 * 32 * 32  # 1024，根据图片大小调整
rf_z_channels = 768  # 与 n_embd 一致
rf_model_depth = 6
rf_model_channel = 1024
rf_num_sampling_steps = 10
rf_grad_checkpointing = False
rf_repeat = 6
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 // gradient_accumulation_steps # how many steps to warm up for
lr_decay_iters = 600000 // gradient_accumulation_steps # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# accelerate + compile settings
mixed_precision = 'auto'  # 'auto' -> bf16 if supported else fp16; or explicitly 'bf16'/'fp16'/'no'
compile = True            # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


# ---------------------------- Accelerate init --------------------------------
# resolve mixed precision
if mixed_precision == 'auto':
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        mp = 'bf16'
    else:
        mp = 'fp16'
else:
    mp = mixed_precision

# We keep the original "global GAS" semantics: when using multiple processes,
# we divide it so that effective tokens/iter stay comparable across world sizes.
# (Same idea as the original DDP code.)
# We need world_size to compute effective GAS, but Accelerator needs GAS for init...
# => compute world_size from env (fallback to 1), then pass effective GAS to Accelerator.
ws_env = int(os.environ.get("WORLD_SIZE", "1"))
assert gradient_accumulation_steps % max(ws_env, 1) == 0, \
    "Please make gradient_accumulation_steps divisible by WORLD_SIZE."
eff_gas = max(1, gradient_accumulation_steps // max(ws_env, 1))

accelerator = Accelerator(gradient_accumulation_steps=eff_gas, mixed_precision=mp)
device = accelerator.device
device_type = device.type  # 'cuda' or 'cpu'

# reproducibility: different streams get different seeds when device_specific=True
set_seed(1337, device_specific=True)

if device_type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ------------------------- IO & misc -------------------------
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

if accelerator.is_main_process:
    os.makedirs(out_dir, exist_ok=True)

# "tokens per iter" metric (comparable to original)
world_size = accelerator.state.num_processes
tokens_per_iter = eff_gas * world_size * batch_size * block_size
if accelerator.is_main_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,} "
          f"(batch_size={batch_size}, eff_gas={eff_gas}, world_size={world_size})")
print(f"tokens per iteration will be: {tokens_per_iter:,}")


# ------------------------- Data -------------------------
data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None # will be set by image loading or meta.pkl

# Helper function to pre-load all images from a directory
def load_images(image_folder_path):
    """Loads all .png grayscale images from a folder, sorts them, and returns a stacked tensor [N, 1, H, W]."""
    image_files = sorted(glob.glob(os.path.join(image_folder_path, '*.png')))
    if not image_files:
        raise FileNotFoundError(f"No PNG images found in {image_folder_path}")

    # 转成单通道灰度并变成 [1, H, W]，像素范围 [0,1]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 统一单通道
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image_tensors = []
    print(f"Loading images from {image_folder_path}...")
    for img_path in image_files:
        with Image.open(img_path) as img:
            # 无论原始是 L / LA / RGB / RGBA，都统一转为单通道
            img = img.convert('L')
            image_tensors.append(transform(img))
    print(f"Loaded {len(image_tensors)} images.")
    return torch.stack(image_tensors)  # [N, 1, H, W]


# Load image dataset if specified, and update vocab size accordingly
image_dataset = None
if dataset == 'chinese_char':
    image_data_path = os.path.join(data_dir, 'images')
    if os.path.isdir(image_data_path):
        image_dataset = load_images(image_data_path)  # 现在是 [N, 1, H, W]
        meta_vocab_size = len(image_dataset)
        if accelerator.is_main_process:
            print(f"Dataset '{dataset}' detected. Vocab size set to {meta_vocab_size} images.")
    else:
        if accelerator.is_main_process:
            print(f"Warning: dataset is '{dataset}' but image directory not found at {image_data_path}")

def get_batch(split):
    # Same "poor man's dataloader" with memmap, keeping RAM usage predictable.
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x_ids = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y_ids = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if image_dataset is not None:
        x = image_dataset[x_ids]
        y = image_dataset[y_ids]
    else:
        # Otherwise, retain the original behavior (use IDs as data)
        x, y = x_ids, y_ids
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.contiguous().pin_memory().to(device, non_blocking=True), y.contiguous().pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# ------------------------- Model & Init -------------------------
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path) and accelerator.is_main_process:
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
# broadcast vocab size to all processes (simple way: every rank re-opens file if exists; harmless)
if meta_vocab_size is None and os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    
# model init (same behavior as original)
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, 
    block_size=block_size, bias=bias, vocab_size=None, dropout=dropout,
    use_rf_loss=use_rf_loss,
    rf_target_channels=rf_target_channels,
    rf_z_channels=rf_z_channels,
    rf_model_depth=rf_model_depth,
    rf_model_channel=rf_model_channel,
    rf_num_sampling_steps=rf_num_sampling_steps,
    rf_grad_checkpointing=rf_grad_checkpointing,
    rf_repeat=rf_repeat
)
if init_from == 'scratch':
    if accelerator.is_main_process:
        print("Initializing a new model from scratch")
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    if accelerator.is_main_process:
        print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix possible unwanted prefix
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    if accelerator.is_main_process:
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
        
# crop down block_size if needed
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# compile BEFORE DDP wrapping for stability
if compile:
    if accelerator.is_main_process:
        print("compiling the model... (PyTorch 2.x)")
    model = torch.compile(model)  # you can set mode="max-autotune" if desired

# Let Accelerate wrap model/optimizer (DDP, autocast, scaler handled inside)
model, optimizer = accelerator.prepare(model, optimizer)


# ------------------------- Eval helper -------------------------
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    # Keep it simple: only rank 0 runs eval to avoid redundant IO/compute.
    # This is acceptable here and avoids extra gather code.
    if not accelerator.is_main_process:
        return {'train': float('nan'), 'val': float('nan')}
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            with accelerator.autocast():
                _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out

# ------------------------- LR schedule -------------------------
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# ------------------------- Wandb -------------------------
if wandb_log and accelerator.is_main_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# ------------------------- Training loop -------------------------
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
while True:
    # set LR for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints

    if iter_num !=0 and iter_num % eval_interval == 0:
        # accelerator.wait_for_everyone()
        losses = estimate_loss()
        if accelerator.is_main_process:
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100
                })
            if (losses['val'] < best_val_loss) or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    # unwrap to get clean state dict
                    raw_state = accelerator.get_state_dict(model)
                    checkpoint = {
                        'model': raw_state,
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    save_path = os.path.join(out_dir, 'ckpt.pt')
                    accelerator.save(checkpoint, save_path)
                    print(f"saving checkpoint to {save_path}")
        # accelerator.wait_for_everyone()
    if iter_num == 0 and eval_only:
        break

    # ---------- forward/backward/update with gradient accumulation ----------
    # We preserve original semantics: micro-step loss is divided by GAS to average gradients.
    total_micro_loss = 0.0
    with accelerator.accumulate(model):
        for micro_step in range(eff_gas):
            X, Y = get_batch('train')  # X: (batch_size, block_size, 1, H, W), Y: (batch_size, block_size)
            with accelerator.autocast():
                logits, loss = model(X, Y)
            total_micro_loss += loss.item()
            loss = loss / eff_gas
            accelerator.backward(loss)

        if grad_clip != 0.0:
            accelerator.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    # timing & logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if (iter_num % log_interval == 0) and accelerator.is_main_process:
        # report the average micro loss over eff_gas
        lossf = total_micro_loss / eff_gas
        if local_iter_num >= 5:
            # mfu uses the unwrapped model
            raw_model = accelerator.unwrap_model(model)
            try:
                mfu = raw_model.estimate_mfu(batch_size * eff_gas, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            except Exception:
                pass
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break