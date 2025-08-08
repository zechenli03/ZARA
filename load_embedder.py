# !/usr/bin/env python
# -*-coding:utf-8 -*-
from mantis.trainer import MantisTrainer
from mantis.architecture import Mantis8M
import numpy as np
import torch
import torch.nn.functional as F
from momentfm import MOMENTPipeline
from tqdm import tqdm


def load_mantis(model_name="paris-noah/Mantis-8M", device="cuda"):
    network = Mantis8M(device=device)
    network = network.from_pretrained(model_name)

    model = MantisTrainer(device=device, network=network)

    return model


def get_mantis_multi_channel_embeddings(model, data_array, device="cuda"):
    data_array = np.array(data_array, dtype=np.float32)
    data_array = data_array.transpose(0, 2, 1)

    num_samples, num_channels, seq_len = data_array.shape
    print(f"num_samples: {num_samples}; Channel num: {num_channels}; seq_len: {seq_len}")

    data_torch = torch.tensor(data_array, dtype=torch.float32, device=device)
    data_torch_scaled = F.interpolate(data_torch, size=512, mode='linear', align_corners=False)

    embeddings = model.transform(data_torch_scaled)
    return embeddings


def load_moment(model_name="AutonLab/MOMENT-1-small", device="cuda"):
    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={'task_name': 'embedding'},
    )

    model.init()
    model.to(device)
    model.eval()


def get_moment_multi_channel_embeddings(data_array, model, batch_size=64, device="cuda"):
    # (N, seq_len, C) → (N, C, seq_len)
    data_array = np.asarray(data_array, dtype=np.float32).transpose(0, 2, 1)
    num_samples, num_channels, seq_len = data_array.shape
    print(f"num_samples: {num_samples}; Channel num: {num_channels}; seq_len: {seq_len}")

    model.to(device).eval()
    all_embeds = []

    with torch.no_grad():
        for start in tqdm(range(0, num_samples, batch_size), desc="Embedding"):
            end = start + batch_size
            batch_np = data_array[start:end]
            batch_t = torch.tensor(batch_np, dtype=torch.float32, device=device)

            outputs = model(x_enc=batch_t)
            embedding = outputs.embeddings
            all_embeds.append(embedding.detach().cpu().numpy())

    embeddings = np.concatenate(all_embeds, axis=0)
    return embeddings


def calculate_mantis_emb(data_np, model, device):
    data_torch = torch.tensor([data_np], dtype=torch.float32, device=device)  # (1, num_channels, seq_len)
    data_torch_scaled = F.interpolate(data_torch, size=512, mode='linear', align_corners=False)

    embeddings_np = model.transform(data_torch_scaled)

    return embeddings_np


def calculate_moment_emb(data_np, model, device):
    data_torch = torch.tensor([data_np], dtype=torch.float32, device=device)  # (1, num_channels, seq_len)

    embeddings_np = model(x_enc=data_torch).embeddings.detach().cpu().numpy()

    return embeddings_np