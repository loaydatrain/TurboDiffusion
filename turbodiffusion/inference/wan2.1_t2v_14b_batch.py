# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified script to run TurboDiffusion 14B on FastVideo prompts

import math
import os

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True

# Hardcoded paths for 14B model
DIT_PATH = "/mnt/fast-disks/hao_lab/loay/TurboDiffusion/checkpoints/TurboWan2.1-T2V-14B-480P.pth"
VAE_PATH = "/mnt/fast-disks/hao_lab/loay/TurboDiffusion/checkpoints/Wan2.1_VAE.pth"
TEXT_ENCODER_PATH = "/mnt/fast-disks/hao_lab/loay/TurboDiffusion/checkpoints/models_t5_umt5-xxl-enc-bf16.pth"
PROMPTS_FILE = "/mnt/fast-disks/hao_lab/loay/FastVideo/turbodiffusion_prompts.txt"
OUTPUT_DIR = "/mnt/fast-disks/hao_lab/loay/TurboDiffusion/outputs_14B"

# Model settings
MODEL = "Wan2.1-14B"
NUM_STEPS = 4
NUM_FRAMES = 81
RESOLUTION = "480p"
ASPECT_RATIO = "16:9"
SIGMA_MAX = 80
SEED = 42
ATTENTION_TYPE = "sla"
SLA_TOPK = 0.1


class Args:
    """Mock args object with hardcoded values."""
    def __init__(self):
        self.dit_path = DIT_PATH
        self.model = MODEL
        self.num_samples = 1
        self.num_steps = NUM_STEPS
        self.sigma_max = SIGMA_MAX
        self.vae_path = VAE_PATH
        self.text_encoder_path = TEXT_ENCODER_PATH
        self.num_frames = NUM_FRAMES
        self.resolution = RESOLUTION
        self.aspect_ratio = ASPECT_RATIO
        self.seed = SEED
        self.attention_type = ATTENTION_TYPE
        self.sla_topk = SLA_TOPK
        self.quant_linear = False
        self.default_norm = False


def load_prompts(path: str) -> list[str]:
    """Load prompts from file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def sanitize_filename(prompt: str, max_len: int = 100) -> str:
    """Create a safe filename from a prompt."""
    # Take first max_len chars and replace problematic characters
    safe = prompt[:max_len].strip()
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n']:
        safe = safe.replace(char, '_')
    return safe


if __name__ == "__main__":
    args = Args()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load prompts
    prompts = load_prompts(PROMPTS_FILE)
    log.info(f"Loaded {len(prompts)} prompts from {PROMPTS_FILE}")
    
    # Load models once
    log.info(f"Loading DiT model from {args.dit_path}")
    net = create_model(dit_path=args.dit_path, args=args).cpu()
    torch.cuda.empty_cache()
    log.success(f"Successfully loaded DiT model.")
    
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
    
    # Generate video for each prompt
    for i, prompt in enumerate(prompts):
        log.info(f"[{i+1}/{len(prompts)}] Generating: {prompt[:60]}...")
        
        # Compute text embedding
        with torch.no_grad():
            text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=prompt).to(**tensor_kwargs)
        clear_umt5_memory()
        
        condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples)}
        
        state_shape = [
            tokenizer.latent_ch,
            tokenizer.get_latent_num_frames(args.num_frames),
            h // tokenizer.spatial_compression_factor,
            w // tokenizer.spatial_compression_factor,
        ]
        
        generator = torch.Generator(device=tensor_kwargs["device"])
        generator.manual_seed(args.seed)
        
        init_noise = torch.randn(
            args.num_samples,
            *state_shape,
            dtype=torch.float32,
            device=tensor_kwargs["device"],
            generator=generator,
        )
        
        # For better visual quality
        mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
        
        t_steps = torch.tensor(
            [math.atan(args.sigma_max), *mid_t, 0],
            dtype=torch.float64,
            device=init_noise.device,
        )
        
        # Convert TrigFlow timesteps to RectifiedFlow
        t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))
        
        # Sampling steps
        x = init_noise.to(torch.float64) * t_steps[0]
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        total_steps = t_steps.shape[0] - 1
        net.cuda()
        for j, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps)):
            with torch.no_grad():
                v_pred = net(x_B_C_T_H_W=x.to(**tensor_kwargs), timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs), **condition).to(
                    torch.float64
                )
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape,
                    dtype=torch.float32,
                    device=tensor_kwargs["device"],
                    generator=generator,
                )
        samples = x.float()
        net.cpu()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            video = tokenizer.decode(samples)
        
        to_show = [(1.0 + video.float().cpu().clamp(-1, 1)) / 2.0]
        to_show = torch.stack(to_show, dim=0)
        
        # Save video
        save_path = os.path.join(OUTPUT_DIR, f"{sanitize_filename(prompt)}.mp4")
        save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), save_path, fps=16)
        log.success(f"Saved to {save_path}")
    
    log.info(f"Done! Generated {len(prompts)} videos to {OUTPUT_DIR}")
