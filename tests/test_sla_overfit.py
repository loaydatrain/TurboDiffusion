#!/usr/bin/env python3
"""
End-to-end overfit test for TurboDiffusion SLA training.

This script verifies TurboDiffusion SLA training by overfitting on a single sample
(the same cat video used in FastVideo tests). After 1000 steps, it generates
an output video for quality comparison with FastVideo.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

# TurboDiffusion paths
TURBODIFFUSION_ROOT = Path(__file__).parent.parent
CHECKPOINT_ROOT = TURBODIFFUSION_ROOT / "assets" / "checkpoints"
DATASET_ROOT = TURBODIFFUSION_ROOT / "assets" / "datasets" / "cat_overfit"
OUTPUT_ROOT = TURBODIFFUSION_ROOT / "outputs" / "sla_overfit_test"

# Cat video info (same as FastVideo test)
CAT_VIDEO_URL = "https://assets.mixkit.co/videos/download/mixkit-pet-owner-playing-with-a-cute-cat-1779.mp4"
CAT_VIDEO_FILENAME = "mixkit-pet-owner-playing-with-a-cute-cat-1779.mp4"
CAT_PROMPT = "A pet owner playing with a cute cat with a straw."

# Training settings
NUM_GPUS = 4
MAX_ITER = 1000


def download_cat_video():
    """Download the cat video if not present."""
    video_dir = TURBODIFFUSION_ROOT / "assets" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / CAT_VIDEO_FILENAME
    
    if video_path.exists():
        print(f"Cat video already exists at {video_path}")
        return video_path
    
    print(f"Downloading cat video to {video_path}...")
    import urllib.request
    urllib.request.urlretrieve(CAT_VIDEO_URL, video_path)
    print("Download complete!")
    return video_path


def create_webdataset(video_path: Path):
    """Create webdataset from the cat video."""
    if DATASET_ROOT.exists() and (DATASET_ROOT / "shard_000000.tar").exists():
        print(f"Webdataset already exists at {DATASET_ROOT}")
        return
    
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        str(TURBODIFFUSION_ROOT / "tests" / "create_overfit_webdataset.py"),
        "--video_path", str(video_path),
        "--prompt", CAT_PROMPT,
        "--output_dir", str(DATASET_ROOT),
        "--vae_path", str(CHECKPOINT_ROOT / "Wan2.1_VAE.pth"),
        "--text_encoder_path", str(CHECKPOINT_ROOT / "models_t5_umt5-xxl-enc-bf16.pth"),
        "--num_frames", "81",
        "--height", "480",
        "--width", "832",
    ]
    
    print(f"Creating webdataset: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TURBODIFFUSION_ROOT / "turbodiffusion")
    subprocess.run(cmd, env=env, check=True)


def ensure_dcp_checkpoint():
    """Ensure DCP checkpoint exists."""
    dcp_path = CHECKPOINT_ROOT / "Wan2.1-T2V-1.3B.dcp"
    if dcp_path.exists():
        print(f"DCP checkpoint already exists at {dcp_path}")
        return
    
    pth_path = CHECKPOINT_ROOT / "Wan2.1-T2V-1.3B.pth"
    if not pth_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pth_path}")
    
    print(f"Converting checkpoint to DCP format...")
    cmd = [
        sys.executable, "-m", "torch.distributed.checkpoint.format_utils",
        "torch_to_dcp", str(pth_path), str(dcp_path)
    ]
    subprocess.run(cmd, check=True)


def run_training():
    """Run SLA training for 1000 steps."""
    # Clean output dir
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TURBODIFFUSION_ROOT / "turbodiffusion")
    env["IMAGINAIRE_OUTPUT_ROOT"] = str(OUTPUT_ROOT)
    env["WANDB_MODE"] = "offline"  # Don't log to wandb for test
    
    cmd = [
        "torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        "--master_port=29515",
        "-m", "scripts.train",
        f"--config=turbodiffusion/rcm/configs/registry_sla.py",
        "--",
        "experiment=wan2pt1_1pt3B_res480p_t2v_SLA",
        f"model.config.teacher_ckpt={CHECKPOINT_ROOT}/Wan2.1-T2V-1.3B.dcp",
        f"model.config.tokenizer.vae_pth={CHECKPOINT_ROOT}/Wan2.1_VAE.pth",
        f"model.config.text_encoder_path={CHECKPOINT_ROOT}/models_t5_umt5-xxl-enc-bf16.pth",
        f"model.config.neg_embed_path={CHECKPOINT_ROOT}/umT5_wan_negative_emb.pt",
        f"dataloader_train.tar_path_pattern={DATASET_ROOT}/shard*.tar",
        f"trainer.max_iter={MAX_ITER}",
        "trainer.logging_iter=10",
        "trainer.callbacks.every_n_sample_reg.every_n=500",
        "trainer.callbacks.every_n_sample_reg.run_at_start=True",
        "trainer.callbacks.every_n_sample_ema.every_n=500",
        "checkpoint.save_iter=500",
        "dataloader_train.batch_size=1",
        "optimizer.lr=1e-5",
        "model.config.fsdp_shard_size=4",
        "job.name=sla_overfit_cat_test",
        "job.group=sla_overfit_test",
    ]
    
    print(f"Running training: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, cwd=str(TURBODIFFUSION_ROOT), check=True)


def generate_output_video():
    """Generate output video using the trained model."""
    # Find the latest checkpoint
    checkpoints = list(OUTPUT_ROOT.glob("**/checkpoint_*.dcp"))
    if not checkpoints:
        print("No checkpoints found, using samples from training callbacks")
        return
    
    latest_ckpt = max(checkpoints, key=lambda x: x.stat().st_mtime)
    print(f"Using checkpoint: {latest_ckpt}")
    
    # The training callbacks should have generated sample videos
    # Look for them in the output directory
    sample_videos = list(OUTPUT_ROOT.glob("**/sample*.mp4"))
    if sample_videos:
        print(f"Found {len(sample_videos)} sample videos:")
        for v in sample_videos:
            print(f"  - {v}")
    else:
        print("No sample videos found from training callbacks")


def verify_outputs():
    """Verify that training produced expected outputs."""
    # Check for any output videos
    sample_videos = list(OUTPUT_ROOT.glob("**/*.mp4"))
    sample_gifs = list(OUTPUT_ROOT.glob("**/*.gif"))
    
    print(f"\n{'='*60}")
    print("Output Verification")
    print(f"{'='*60}")
    print(f"Found {len(sample_videos)} MP4 files")
    print(f"Found {len(sample_gifs)} GIF files")
    
    if sample_videos or sample_gifs:
        print("\nOutput files:")
        for f in sample_videos + sample_gifs:
            print(f"  - {f.relative_to(OUTPUT_ROOT)}")
        print("\nSUCCESS: Training completed with video outputs!")
        return True
    else:
        print("\nWARNING: No video outputs found")
        return False


def main():
    print("="*60)
    print("TurboDiffusion SLA Overfit Test")
    print("="*60)
    print(f"Checkpoint root: {CHECKPOINT_ROOT}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print()
    
    # Step 1: Download cat video
    print("\n[1/5] Downloading cat video...")
    video_path = download_cat_video()
    
    # Step 2: Ensure DCP checkpoint
    print("\n[2/5] Checking DCP checkpoint...")
    ensure_dcp_checkpoint()
    
    # Step 3: Create webdataset
    print("\n[3/5] Creating webdataset...")
    create_webdataset(video_path)
    
    # Step 4: Run training
    print("\n[4/5] Running SLA training...")
    run_training()
    
    # Step 5: Verify outputs
    print("\n[5/5] Verifying outputs...")
    success = verify_outputs()
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS: TurboDiffusion SLA overfit test completed!")
        print("="*60)
        print(f"\nOutput videos are in: {OUTPUT_ROOT}")
        print("Compare these with FastVideo outputs to verify training behavior.")
    else:
        print("\n" + "="*60)
        print("WARNING: Test completed but no output videos found")
        print("="*60)


if __name__ == "__main__":
    main()
