# -------------------------------------------------------------------------
# FORCE PATHS: Add NVIDIA libraries from Conda to LD_LIBRARY_PATH
# -------------------------------------------------------------------------
echo "Configuring LD_LIBRARY_PATH..."

# 1. Get the site-packages folder
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# 2. Add specific NVIDIA library paths
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/nccl/lib:$LD_LIBRARY_PATH

# 3. Add Conda lib just in case
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH is now set to:"
echo $LD_LIBRARY_PATH
# -------------------------------------------------------------------------

WORKDIR="/mnt/weka/home/hao.zhang/l0ay/TurboDiffusion"
cd $WORKDIR
export PYTHONPATH=turbodiffusion

# the "IMAGINAIRE_OUTPUT_ROOT" environment variable is the path to save experiment output files
export IMAGINAIRE_OUTPUT_ROOT=${WORKDIR}/outputs
CHECKPOINT_ROOT=${WORKDIR}/assets/checkpoints
DATASET_ROOT=${WORKDIR}/assets/datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K

# your Wandb information
export WANDB_API_KEY=585ae8dab7a9cc8141cf080989f3c836a18a29e9
export WANDB_ENTITY=loayrashid-university-of-california-san-diego

registry=registry_sla
experiment=wan2pt1_1pt3B_res480p_t2v_SLA

torchrun --nproc_per_node=4 \
    -m scripts.train --config=turbodiffusion/rcm/configs/${registry}.py -- experiment=${experiment} \
        model.config.teacher_ckpt=${CHECKPOINT_ROOT}/Wan2.1-T2V-1.3B.dcp \
        model.config.tokenizer.vae_pth=${CHECKPOINT_ROOT}/Wan2.1_VAE.pth \
        model.config.text_encoder_path=${CHECKPOINT_ROOT}/models_t5_umt5-xxl-enc-bf16.pth \
        model.config.neg_embed_path=${CHECKPOINT_ROOT}/umT5_wan_negative_emb.pt \
        dataloader_train.tar_path_pattern=${DATASET_ROOT}/shard*.tar