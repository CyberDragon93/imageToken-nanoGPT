"""
This file contains code adapted from the clipscore repository:
https://github.com/jmhessel/clipscore
"""

import clip
import numpy as np
import sklearn.preprocessing
import torch
import tqdm
import warnings

from packaging import version
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose(
            [
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {"image": image}

    def __len__(self):
        return len(self.data)


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix="A photo depicts"):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != " ":
            self.prefix += " "

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {"caption": c_data}

    def __len__(self):
        return len(self.data)


def extract_feats(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b["image"].to(device)
            if device == "cuda":
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def extract_text_feats(captions, model, device, batch_size=256, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b["caption"].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def get_clip_score(model, images_feats, candidates, device, w=2.5):
    """
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    """

    candidates = extract_text_feats(candidates, model, device)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse("1.21"):
        images_feats = sklearn.preprocessing.normalize(images_feats, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            "due to a numerical instability, new numpy normalization is slightly different than paper results. "
            "to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3."
        )
        images_feats = images_feats / np.sqrt(
            np.sum(images_feats**2, axis=1, keepdims=True)
        )
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(images_feats * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates


def calculate_clipscore(
    image_list,
    text_list,
    clip_model="ViT-B/32",
    batch_size=64,
    num_workers=8,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, transform = clip.load(clip_model, device=device)
    model.eval()

    image_feats = extract_feats(
        image_list, model, device, batch_size=batch_size, num_workers=num_workers
    )

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feats, text_list, device, w=1.0
    )
    clip_score_avg = np.mean(per_instance_image_text)
    return clip_score_avg
