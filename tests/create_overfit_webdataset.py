#!/usr/bin/env python3
"""
Create webdataset from a single video for overfitting test.

This script encodes a video with WanVAE and text with umT5 encoder,
then packages them into webdataset format for TurboDiffusion training.
"""
import argparse
import os
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# Add TurboDiffusion to path
SCRIPT_DIR = Path(__file__).parent
TURBODIFFUSION_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(TURBODIFFUSION_ROOT / "turbodiffusion"))


def load_video_frames(video_path: str, num_frames: int = 81, height: int = 480, width: int = 832) -> torch.Tensor:
    """Load and preprocess video frames."""
    import decord
    from decord import VideoReader, cpu
    
    decord.bridge.set_bridge("torch")
    
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    # Sample frames uniformly
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Repeat frames if video is too short
        indices = np.arange(num_frames) % total_frames
    
    frames = vr.get_batch(indices)  # [T, H, W, C]
    frames = frames.permute(0, 3, 1, 2).float()  # [T, C, H, W]
    
    # Resize to target resolution
    frames = F.interpolate(frames, size=(height, width), mode='bilinear', align_corners=False)
    
    # Normalize to [-1, 1]
    frames = frames / 127.5 - 1.0
    
    return frames  # [T, C, H, W]


def encode_video_with_vae(video: torch.Tensor, vae_path: str, device: str = "cuda") -> torch.Tensor:
    """Encode video frames with WanVAE."""
    from rcm.tokenizers.wan2pt1 import WanVAE
    
    print(f"Loading VAE from {vae_path}...")
    vae = WanVAE(z_dim=16, vae_pth=vae_path, device=device)
    
    # video: [T, C, H, W] -> [1, C, T, H, W]
    video = video.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
    print(f"Video tensor shape: {video.shape}, dtype: {video.dtype}")
    
    with torch.no_grad():
        # Encode video to latents (WanVAE.encode already normalizes)
        latents = vae.encode(video)  # [1, z_dim, t, h, w]
    
    print(f"Encoded latents shape: {latents.shape}")
    print(f"Latents stats: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
    
    del vae
    torch.cuda.empty_cache()
    
    # Remove batch dimension and squeeze for webdataset format
    return latents.squeeze(0).cpu()  # [z_dim, t, h, w]


def encode_prompt_with_t5(prompt: str, text_encoder_path: str, device: str = "cuda") -> torch.Tensor:
    """Encode prompt with umT5 text encoder."""
    from rcm.utils.umt5 import UMT5EncoderModel
    
    print(f"Loading T5 encoder from {text_encoder_path}...")
    text_encoder = UMT5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=text_encoder_path,
    )
    
    with torch.no_grad():
        embeddings = text_encoder([prompt], device=device)
    
    print(f"Encoded embeddings shape: {embeddings.shape}")
    
    del text_encoder
    torch.cuda.empty_cache()
    
    # Remove batch dimension for webdataset format
    return embeddings.squeeze(0).cpu()  # [seq_len, dim]


def create_webdataset_shard(
    output_path: str,
    latents: torch.Tensor,
    embeddings: torch.Tensor,
    prompt: str,
    sample_key: str = "000000",
):
    """Create a webdataset tar file with the encoded data."""
    print(f"Creating webdataset shard at {output_path}...")
    
    with tarfile.open(output_path, "w") as tar:
        # Add latent.pt
        latent_buffer = BytesIO()
        torch.save(latents, latent_buffer)
        latent_buffer.seek(0)
        latent_info = tarfile.TarInfo(name=f"{sample_key}.latent.pt")
        latent_info.size = len(latent_buffer.getvalue())
        tar.addfile(latent_info, latent_buffer)
        
        # Add embed.pt
        embed_buffer = BytesIO()
        torch.save(embeddings, embed_buffer)
        embed_buffer.seek(0)
        embed_info = tarfile.TarInfo(name=f"{sample_key}.embed.pt")
        embed_info.size = len(embed_buffer.getvalue())
        tar.addfile(embed_info, embed_buffer)
        
        # Add prompt.txt
        prompt_bytes = prompt.encode("utf-8")
        prompt_buffer = BytesIO(prompt_bytes)
        prompt_info = tarfile.TarInfo(name=f"{sample_key}.prompt.txt")
        prompt_info.size = len(prompt_bytes)
        tar.addfile(prompt_info, prompt_buffer)
    
    print(f"Created webdataset shard with 1 sample")
    print(f"  - latent.pt: {latents.shape}")
    print(f"  - embed.pt: {embeddings.shape}")
    print(f"  - prompt.txt: {len(prompt)} chars")


def main():
    parser = argparse.ArgumentParser(description="Create webdataset from video for overfitting")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for the video")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for webdataset")
    parser.add_argument("--vae_path", type=str, default="assets/checkpoints/Wan2.1_VAE.pth")
    parser.add_argument("--text_encoder_path", type=str, default="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and encode video
    print(f"Loading video from {args.video_path}...")
    video = load_video_frames(args.video_path, args.num_frames, args.height, args.width)
    print(f"Video shape: {video.shape}")
    
    # Encode with VAE
    print("\nEncoding video with VAE...")
    latents = encode_video_with_vae(video, args.vae_path, args.device)
    print(f"Latents shape: {latents.shape}")
    
    # Encode prompt
    print("\nEncoding prompt with T5...")
    embeddings = encode_prompt_with_t5(args.prompt, args.text_encoder_path, args.device)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Create webdataset shard
    print("\nCreating webdataset...")
    output_path = os.path.join(args.output_dir, "shard_000000.tar")
    create_webdataset_shard(output_path, latents, embeddings, args.prompt)
    
    print(f"\nDone! Webdataset created at {output_path}")


if __name__ == "__main__":
    main()
