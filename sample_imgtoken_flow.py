"""
Sample from a trained model (image->text autoregressive)
"""
import os
import glob
import pickle
from contextlib import nullcontext

import torch
import tiktoken
from PIL import Image
from torchvision import transforms

from model_imgtoken_flow import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'  # 'resume' or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'       # ignored if init_from is not 'resume'
start = "\n"          # or "FILE:prompt.txt"
num_samples = 10
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ------------------------- helpers -------------------------
def load_images(image_folder_path):
    """Loads all .png grayscale images as a bank tensor [V, 1, H, W]."""
    image_files = sorted(glob.glob(os.path.join(image_folder_path, '*.png')))
    if not image_files:
        raise FileNotFoundError(f"No PNG images found in {image_folder_path}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    image_tensors = []
    print(f"Loading images from {image_folder_path}...")
    for img_path in image_files:
        with Image.open(img_path) as img:
            img = img.convert('L')
            image_tensors.append(transform(img))
    print(f"Loaded {len(image_tensors)} images.")
    return torch.stack(image_tensors)  # [V, 1, H, W]

# ------------------------- model -------------------------
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# ------------------------- tokenizer / meta -------------------------
load_meta = False
meta_path = None
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings (text-only fallback)...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# ------------------------- start prompt -------------------------
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

start = "秋高气爽时，苹果采摘季。伴着美好的“丰”景和醇厚的果香，记者来到陕西延安，"

# ------------------------- vision or text path -------------------------
use_vision = getattr(model.config, 'use_vision_encoder', False)

if use_vision:
    # ===== Vision-conditioned generation =====
    # 1) figure dataset/images folder
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        dataset_name = checkpoint['config']['dataset']
    else:
        # fallback: try a common name; adjust if needed
        dataset_name = 'chinese_char'
    image_data_path = os.path.join('data', dataset_name, 'images')

    # 2) build image bank: [V, 1, H, W] (can stay on CPU; generate() handles per-step fetch)
    image_bank = load_images(image_data_path)
    V = image_bank.shape[0]

    # 3) sanity check vocab size vs image bank size
    if hasattr(model.config, 'vocab_size') and model.config.vocab_size != V:
        print(f"Warning: image_bank size ({V}) != model vocab_size ({model.config.vocab_size})")

    # 4) build start_ids -> initial image sequence
    if load_meta:
        start_ids_list = encode(start)
        if len(start_ids_list) == 0:
            start_ids_list = [0]
    else:
        start_ids_list = [0]

    start_ids = torch.tensor(start_ids_list, dtype=torch.long, device=device)[None, ...]  # [1, T0]
    x0 = image_bank[start_ids.to('cpu')]  # [1, T0, 1, H, W]
    # e([1, 11, 1, 32, 32])
    print(f"start_ids_list: {start_ids_list}, x0.shape: {x0.shape}")

    with torch.no_grad(), ctx:
        for k in range(num_samples):
            # idx, max_new_tokens, image_bank=None, return_images=False, cfg=1.0
            y_ids = model.generate(
                x0.clone().to(device),
                max_new_tokens,
                image_bank=image_bank,   # [V, 1, H, W] (on CPU is fine)
            )
            print(decode(y_ids[0].tolist()))
            print(y_ids.shape)
            print('---------------')

else:
    raise ValueError("GPT model without RF loss or Vision Encoder is not supported here.")
    # ===== Pure text generation (original path) =====
    # start_ids_list = encode(start)
    # x = torch.tensor(start_ids_list, dtype=torch.long, device=device)[None, ...]
    # with torch.no_grad(), ctx:
    #     for k in range(num_samples):
    #         y = model.generate(x.clone(), max_new_tokens, temperature=temperature, top_k=top_k)
    #         print(decode(y[0].tolist()))
    #         print('---------------')