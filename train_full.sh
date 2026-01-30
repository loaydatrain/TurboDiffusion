WORKDIR="/mnt/fast-disks/hao_lab/loay/TurboDiffusion"
cd $WORKDIR
export PYTHONPATH=turbodiffusion

# the "IMAGINAIRE_OUTPUT_ROOT" environment variable is the path to save experiment output files
export IMAGINAIRE_OUTPUT_ROOT=${WORKDIR}/outputs
CHECKPOINT_ROOT=${WORKDIR}/assets/checkpoints
DATASET_ROOT=${WORKDIR}/assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K

# your Wandb information
export WANDB_API_KEY=xxx
export WANDB_ENTITY=xxx

registry=registry_sla
experiment=wan2pt1_1pt3B_res480p_t2v_SLA

torchrun --nproc_per_node=4 \
    -m scripts.train --config=turbodiffusion/rcm/configs/${registry}.py -- experiment=${experiment} \
        model.config.teacher_ckpt=${CHECKPOINT_ROOT}/Wan2.1-T2V-1.3B.dcp \
        model.config.tokenizer.vae_pth=${CHECKPOINT_ROOT}/Wan2.1_VAE.pth \
        model.config.text_encoder_path=${CHECKPOINT_ROOT}/models_t5_umt5-xxl-enc-bf16.pth \
        model.config.neg_embed_path=${CHECKPOINT_ROOT}/umT5_wan_negative_emb.pt \
        dataloader_train.tar_path_pattern=${DATASET_ROOT}/shard*.tar