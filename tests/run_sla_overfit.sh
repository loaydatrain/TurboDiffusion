#!/bin/bash
# TurboDiffusion SLA Overfit Training Script
# Uses the same cat video as FastVideo test for comparison

set -e

# Configuration
WORKDIR="/mnt/fast-disks/hao_lab/loay/TurboDiffusion"
CHECKPOINT_ROOT="${WORKDIR}/assets/checkpoints"
DATASET_ROOT="${WORKDIR}/assets/datasets/cat_overfit_4shards"

# Training parameters
NUM_GPUS=4
MAX_ITER=1000
BATCH_SIZE=1
LEARNING_RATE=1e-5
LOGGING_ITER=10
CHECKPOINT_ITER=500
SAMPLE_ITER=100
MASTER_PORT=29517

cd "$WORKDIR"

# Environment setup
export PYTHONPATH=turbodiffusion
export IMAGINAIRE_OUTPUT_ROOT=outputs
export WANDB_MODE=offline

echo "Starting TurboDiffusion SLA overfit training..."
echo "  Checkpoints: ${CHECKPOINT_ROOT}"
echo "  Dataset: ${DATASET_ROOT}"
echo "  Max iterations: ${MAX_ITER}"
echo "  GPUs: ${NUM_GPUS}"

torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
    -m scripts.train \
    --config=turbodiffusion/rcm/configs/registry_sla.py \
    -- \
    experiment=wan2pt1_1pt3B_res480p_t2v_SLA \
    model.config.teacher_ckpt=${CHECKPOINT_ROOT}/Wan2.1-T2V-1.3B.dcp \
    model.config.tokenizer.vae_pth=${CHECKPOINT_ROOT}/Wan2.1_VAE.pth \
    model.config.text_encoder_path=${CHECKPOINT_ROOT}/models_t5_umt5-xxl-enc-bf16.pth \
    model.config.neg_embed_path=${CHECKPOINT_ROOT}/umT5_wan_negative_emb.pt \
    dataloader_train.tar_path_pattern=${DATASET_ROOT}/shard*.tar \
    dataloader_train.num_workers=1 \
    dataloader_train.shuffle_buffer=1 \
    trainer.max_iter=${MAX_ITER} \
    trainer.logging_iter=${LOGGING_ITER} \
    trainer.callbacks.every_n_sample_reg.every_n=${SAMPLE_ITER} \
    trainer.callbacks.every_n_sample_reg.run_at_start=True \
    trainer.callbacks.every_n_sample_ema.every_n=${SAMPLE_ITER} \
    checkpoint.save_iter=${CHECKPOINT_ITER} \
    dataloader_train.batch_size=${BATCH_SIZE} \
    optimizer.lr=${LEARNING_RATE} \
    model.config.fsdp_shard_size=${NUM_GPUS} \
    job.name=sla_overfit_cat_test \
    job.group=sla_overfit_test

echo "Training complete!"
echo "Outputs saved to: ${WORKDIR}/outputs/sla/sla_overfit_test/sla_overfit_cat_test/"
