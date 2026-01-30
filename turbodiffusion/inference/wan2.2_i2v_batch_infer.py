# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TurboDiffusion batch I2V inference script.

Runs I2V generation on all images in assets/i2v_inputs/ using prompts from prompts.txt.

Usage:
    python turbodiffusion/inference/wan2.2_i2v_batch_infer.py \
        --high_noise_model_path checkpoints/TurboWan2.2-I2V-A14B-high-720P.pth \
        --low_noise_model_path checkpoints/TurboWan2.2-I2V-A14B-low-720P.pth \
        --output_dir output/i2v_batch
"""

import argparse
import math
import os
from pathlib import Path

import torch
from einops import rearrange, repeat
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as T
import numpy as np

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True

# Default paths
ASSETS_DIR = Path(__file__).parent.parent.parent / "assets" / "i2v_inputs"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboDiffusion batch I2V inference")
    parser.add_argument("--input_dir", type=str, default=str(ASSETS_DIR),
                        help="Directory containing input images and prompts.txt")
    parser.add_argument("--high_noise_model_path", type=str, required=True,
                        help="Path to the high-noise model")
    parser.add_argument("--low_noise_model_path", type=str, required=True,
                        help="Path to the low-noise model")
    parser.add_argument("--boundary", type=float, default=0.9,
                        help="Timestep boundary for switching from high to low noise model")
    parser.add_argument("--model", choices=["Wan2.2-A14B"], default="Wan2.2-A14B",
                        help="Model to use")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate per image")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4,
                        help="1~4 for timestep-distilled inference")
    parser.add_argument("--sigma_max", type=float, default=200,
                        help="Initial sigma for rCM")
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth",
                        help="Path to the Wan2.1 VAE")
    parser.add_argument("--text_encoder_path", type=str,
                        default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                        help="Path to the umT5 text encoder")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames to generate")
    parser.add_argument("--resolution", default="720p", type=str,
                        help="Resolution of the generated output")
    parser.add_argument("--aspect_ratio", default="16:9", type=str,
                        help="Aspect ratio of the generated output (width:height)")
    parser.add_argument("--adaptive_resolution", action="store_true",
                        help="Adapt output resolution to input image's aspect ratio")
    parser.add_argument("--ode", action="store_true",
                        help="Use ODE for sampling (sharper but less robust than SDE)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="output/i2v_batch",
                        help="Directory to save generated videos")
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"],
                        default="sla", help="Type of attention mechanism to use")
    parser.add_argument("--sla_topk", type=float, default=0.1,
                        help="Top-k ratio for SLA/SageSLA attention")
    parser.add_argument("--quant_linear", action="store_true",
                        help="Whether to replace Linear layers with quantized versions")
    parser.add_argument("--default_norm", action="store_true",
                        help="Whether to replace LayerNorm/RMSNorm layers with faster versions")
    return parser.parse_args()


def load_prompts(prompts_file: Path) -> list[str]:
    """Load prompts from prompts.txt file."""
    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    return prompts


def get_image_files(input_dir: Path) -> list[Path]:
    """Get sorted list of input images (i2v_input_*.jpg)."""
    images = sorted(input_dir.glob("i2v_input_*.jpg"))
    return images


def generate_video(
    image_path: Path,
    prompt: str,
    args: argparse.Namespace,
    high_noise_model,
    low_noise_model,
    tokenizer,
    output_path: Path,
) -> None:
    """Generate a single video from image and prompt."""
    
    # Get text embedding
    with torch.no_grad():
        text_emb = get_umt5_embedding(
            checkpoint_path=args.text_encoder_path,
            prompts=prompt
        ).to(**tensor_kwargs)
    
    # Load and preprocess image
    input_image = Image.open(image_path).convert("RGB")
    
    if args.adaptive_resolution:
        base_w, base_h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
        max_resolution_area = base_w * base_h
        orig_w, orig_h = input_image.size
        image_aspect_ratio = orig_h / orig_w
        ideal_w = np.sqrt(max_resolution_area / image_aspect_ratio)
        ideal_h = np.sqrt(max_resolution_area * image_aspect_ratio)
        stride = tokenizer.spatial_compression_factor * 2
        lat_h = round(ideal_h / stride)
        lat_w = round(ideal_w / stride)
        h = lat_h * stride
        w = lat_w * stride
    else:
        w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
    
    F = args.num_frames
    lat_h = h // tokenizer.spatial_compression_factor
    lat_w = w // tokenizer.spatial_compression_factor
    lat_t = tokenizer.get_latent_num_frames(F)

    image_transforms = T.Compose([
        T.ToImage(),
        T.Resize(size=(h, w), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image_tensor = image_transforms(input_image).unsqueeze(0).to(
        device=tensor_kwargs["device"], dtype=torch.float32
    )

    with torch.no_grad():
        frames_to_encode = torch.cat([
            image_tensor.unsqueeze(2),
            torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device)
        ], dim=2)
        encoded_latents = tokenizer.encode(frames_to_encode)
        del frames_to_encode
        torch.cuda.empty_cache()

    msk = torch.zeros(1, 4, lat_t, lat_h, lat_w, **tensor_kwargs)
    msk[:, :, 0, :, :] = 1.0
    y = torch.cat([msk, encoded_latents.to(**tensor_kwargs)], dim=1)
    y = y.repeat(args.num_samples, 1, 1, 1, 1)

    condition = {
        "crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples),
        "y_B_C_T_H_W": y
    }

    state_shape = [tokenizer.latent_ch, lat_t, lat_h, lat_w]
    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    init_noise = torch.randn(
        args.num_samples, *state_shape,
        dtype=torch.float32, device=tensor_kwargs["device"], generator=generator,
    )

    mid_t = [1.5, 1.4, 1.0][:args.num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64, device=init_noise.device,
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    
    high_noise_model.cuda()
    net = high_noise_model
    switched = False
    
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        if t_cur.item() < args.boundary and not switched:
            high_noise_model.cpu()
            torch.cuda.empty_cache()
            low_noise_model.cuda()
            net = low_noise_model
            switched = True
        
        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition
            ).to(torch.float64)
            
            if args.ode:
                x = x - (t_cur - t_next) * v_pred
            else:
                x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                    *x.shape, dtype=torch.float32,
                    device=tensor_kwargs["device"], generator=generator,
                )
    
    samples = x.float()
    low_noise_model.cpu()
    torch.cuda.empty_cache()

    with torch.no_grad():
        video = tokenizer.decode(samples)

    to_show = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0
    save_image_or_video(rearrange(to_show, "b c t h w -> c t h (b w)"), str(output_path), fps=16)


if __name__ == "__main__":
    args = parse_arguments()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts and get images
    prompts = load_prompts(input_dir / "prompts.txt")
    images = get_image_files(input_dir)
    
    log.info(f"Found {len(images)} images and {len(prompts)} prompts")
    
    if len(images) != len(prompts):
        log.warning(f"Number of images ({len(images)}) != prompts ({len(prompts)})")
    
    # Load models once
    log.info("Loading DiT models...")
    high_noise_model = create_model(dit_path=args.high_noise_model_path, args=args).cpu()
    torch.cuda.empty_cache()
    low_noise_model = create_model(dit_path=args.low_noise_model_path, args=args).cpu()
    torch.cuda.empty_cache()
    log.success("Successfully loaded DiT models.")
    
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    
    # Clear text encoder memory before batch processing
    clear_umt5_memory()
    
    # Process each image
    for i, (image_path, prompt) in enumerate(tqdm(zip(images, prompts), total=min(len(images), len(prompts)), desc="Generating videos")):
        output_path = output_dir / f"i2v_output_{i:02d}.mp4"
        
        log.info(f"\n[{i+1}/{len(images)}] Processing: {image_path.name}")
        log.info(f"Prompt: {prompt[:80]}...")
        
        generate_video(
            image_path=image_path,
            prompt=prompt,
            args=args,
            high_noise_model=high_noise_model,
            low_noise_model=low_noise_model,
            tokenizer=tokenizer,
            output_path=output_path,
        )
        
        log.success(f"Saved: {output_path}")
        
        # Clear text encoder memory between generations
        clear_umt5_memory()
    
    log.success(f"\n✓ Done! Generated {min(len(images), len(prompts))} videos to {output_dir}")
